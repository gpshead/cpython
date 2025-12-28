/*
 * SIMD-accelerated base64 encoding/decoding
 *
 * AVX-512 VBMI implementation - uses vpermi2b for direct 64-byte table lookup
 */

#ifndef BASE64_SIMD_H
#define BASE64_SIMD_H

#include "Python.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#define BASE64_X86_64
#endif

/*
 * CPU feature detection
 */
#ifdef BASE64_X86_64

static int _base64_cpu_checked = 0;
static int _base64_has_avx512vbmi = 0;

#ifdef _MSC_VER
#include <intrin.h>
#define CPUID(leaf, a, b, c, d) do { \
    int regs[4]; __cpuidex(regs, leaf, 0); \
    a = regs[0]; b = regs[1]; c = regs[2]; d = regs[3]; \
} while(0)
#else
#include <cpuid.h>
#define CPUID(leaf, a, b, c, d) __cpuid_count(leaf, 0, a, b, c, d)
#endif

static inline void
_base64_detect_cpu(void)
{
    if (_base64_cpu_checked) return;

    int eax, ebx, ecx, edx;
    CPUID(0, eax, ebx, ecx, edx);

    if (eax >= 7) {
        CPUID(7, eax, ebx, ecx, edx);
        /* AVX-512 VBMI is bit 1 of ECX (leaf 7) */
        _base64_has_avx512vbmi = (ecx >> 1) & 1;
    }
    _base64_cpu_checked = 1;
}

static inline int
base64_has_avx512vbmi(void)
{
    _base64_detect_cpu();
    return _base64_has_avx512vbmi;
}

#else

static inline int base64_has_avx512vbmi(void) { return 0; }

#endif /* BASE64_X86_64 */


/*
 * AVX-512 VBMI Implementation
 *
 * Key insight: vpermi2b can look up 64 bytes from a 64-byte table in one instruction!
 * This is perfect for base64's 64-character alphabet.
 *
 * Encoding: 48 input bytes -> 64 output chars
 * Decoding: 64 input chars -> 48 output bytes
 */
#ifdef BASE64_X86_64

#include <immintrin.h>

/*
 * Use target attributes on GCC/Clang to compile AVX-512 code on any x86-64.
 * MSVC doesn't support function-level target attributes; it requires /arch:AVX512.
 */
#if defined(__GNUC__) || defined(__clang__)
#define BASE64_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vbmi")))
#define BASE64_HAS_AVX512_COMPILED 1
#elif defined(_MSC_VER) && defined(__AVX512VBMI__)
#define BASE64_AVX512_TARGET
#define BASE64_HAS_AVX512_COMPILED 1
#else
#define BASE64_HAS_AVX512_COMPILED 0
#endif

#if BASE64_HAS_AVX512_COMPILED

/* The base64 encoding table as a 512-bit vector */
static const char _b64_table[64] __attribute__((aligned(64))) =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/*
 * Encode 48 bytes to 64 base64 characters using AVX-512 VBMI
 */
BASE64_AVX512_TARGET
static inline void
base64_encode_48_avx512(const uint8_t *in, uint8_t *out)
{
    /* Load the base64 table into a zmm register */
    __m512i table = _mm512_load_si512((const __m512i *)_b64_table);

    /* Load 48 input bytes (load 64, we'll only use 48) */
    __m512i input = _mm512_loadu_si512((const __m512i *)in);

    /*
     * Reshuffle input bytes to prepare for 6-bit extraction.
     *
     * Input: 48 bytes = 16 triplets [b0,b1,b2], [b3,b4,b5], ...
     * Each triplet produces 4 sextets (6-bit values).
     *
     * We arrange bytes so each 32-bit word contains one triplet:
     * [0, b0, b1, b2] -> can extract 4 sextets from 24 bits
     */
    /*
     * Reshuffle for the multiply-shift trick using cross-lane byte permute.
     *
     * For each triplet (b0, b1, b2) at input indices [i, i+1, i+2],
     * output 4 bytes: [b1, b0, b2, b1] for the multiply-mask extraction.
     *
     * We use _mm512_permutexvar_epi8 (vpermb) which can select any byte
     * from the 64-byte input vector, unlike vpshufb which is lane-local.
     *
     * 16 triplets (48 bytes) produce 64 output bytes.
     * Triplet n uses input bytes [3n, 3n+1, 3n+2].
     * Output bytes [4n, 4n+1, 4n+2, 4n+3] = [b1, b0, b2, b1] = [3n+1, 3n, 3n+2, 3n+1]
     *
     * _mm512_set_epi8 args go from high byte (63) to low byte (0).
     */
    /*
     * For triplet n (input bytes 3n, 3n+1, 3n+2 = b0, b1, b2):
     * Output bytes 4n, 4n+1, 4n+2, 4n+3 should contain [b1, b0, b2, b1]
     * = indices [3n+1, 3n, 3n+2, 3n+1]
     *
     * In set_epi8(a63, ..., a0) order (high to low byte):
     * a4n+3=3n+1, a4n+2=3n+2, a4n+1=3n, a4n=3n+1
     * Written left to right: 3n+1, 3n+2, 3n, 3n+1
     */
    const __m512i shuf_input = _mm512_set_epi8(
        /* Triplets 15-12: T15=[45,46,47], T14=[42,43,44], T13=[39,40,41], T12=[36,37,38] */
        46, 47, 45, 46,   43, 44, 42, 43,   40, 41, 39, 40,   37, 38, 36, 37,
        /* Triplets 11-8: T11=[33,34,35], T10=[30,31,32], T9=[27,28,29], T8=[24,25,26] */
        34, 35, 33, 34,   31, 32, 30, 31,   28, 29, 27, 28,   25, 26, 24, 25,
        /* Triplets 7-4: T7=[21,22,23], T6=[18,19,20], T5=[15,16,17], T4=[12,13,14] */
        22, 23, 21, 22,   19, 20, 18, 19,   16, 17, 15, 16,   13, 14, 12, 13,
        /* Triplets 3-0: T3=[9,10,11], T2=[6,7,8], T1=[3,4,5], T0=[0,1,2] */
        10, 11,  9, 10,    7,  8,  6,  7,    4,  5,  3,  4,    1,  2,  0,  1
    );

    /* Use vpermb for cross-lane byte shuffle */
    __m512i triplets = _mm512_permutexvar_epi8(shuf_input, input);

    /*
     * Now each 32-bit word has: [0, b0, b1, b2] in big-endian order
     * As a little-endian 32-bit value: (b0 << 16) | (b1 << 8) | b2
     *
     * Extract 4 sextets from each 32-bit word:
     *   sextet0 = (word >> 18) & 0x3F  = b0[7:2]
     *   sextet1 = (word >> 12) & 0x3F  = b0[1:0]:b1[7:4]
     *   sextet2 = (word >> 6) & 0x3F   = b1[3:0]:b2[7:6]
     *   sextet3 = word & 0x3F          = b2[5:0]
     *
     * Use multishift: vpmultishiftqb can extract 8 different 8-bit fields
     * from each 64-bit element. But we'll use the mask-and-multiply trick.
     */

    /* Method: use AND + multiply to extract and position the 6-bit values */

    /* For each 32-bit word, extract bits into 4 separate bytes */
    /* After shuffle, word = (b0 << 16) | (b1 << 8) | b2 (as 24-bit value in low 24 bits) */

    /* Actually on little-endian x86, after the shuffle the bytes are:
     * byte0 = b2, byte1 = b1, byte2 = b0, byte3 = 0
     * So as a 32-bit value: (0 << 24) | (b0 << 16) | (b1 << 8) | b2
     */

    /* Use the standard bit-extraction technique */
    __m512i t0 = _mm512_and_si512(triplets, _mm512_set1_epi32(0x0FC0FC00));
    __m512i t1 = _mm512_and_si512(triplets, _mm512_set1_epi32(0x003F03F0));

    /* Multiply to shift bits into position */
    /* t0: has bits we need in positions that mulhi will shift down */
    /* t1: has bits we need in positions that mullo will shift up */
    t0 = _mm512_mulhi_epu16(t0, _mm512_set1_epi32(0x04000040));
    t1 = _mm512_mullo_epi16(t1, _mm512_set1_epi32(0x01000010));

    /* Combine to get 6-bit indices in each byte */
    __m512i indices = _mm512_or_si512(t0, t1);

    /*
     * Now use vpermi2b to look up all 64 characters at once!
     * indices contains 64 bytes, each with a 6-bit index (0-63)
     * table contains the 64-character base64 alphabet
     * vpermi2b selects bytes from the concatenation of two sources
     */

    /* vpermi2b: for each byte in indices, look up from table */
    /* _mm512_permutexvar_epi8 does exactly this */
    __m512i result = _mm512_permutexvar_epi8(indices, table);

    /* Store 64 output characters */
    _mm512_storeu_si512((__m512i *)out, result);
}


/*
 * Build the decode lookup table.
 * Maps ASCII character to 6-bit value (0-63), or 0xFF for invalid.
 * Only first 128 entries needed (ASCII).
 */
static const int8_t _b64_decode_table[128] __attribute__((aligned(64))) = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  /* 0-15 */
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  /* 16-31 */
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,  /* 32-47: + is 43, / is 47 */
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,  /* 48-63: 0-9 */
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  /* 64-79: A-O */
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,  /* 80-95: P-Z */
    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,  /* 96-111: a-o */
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,  /* 112-127: p-z */
};

/*
 * Decode 64 base64 characters to 48 bytes using AVX-512 VBMI
 * Returns 48 on success, -1 if padding found or invalid character
 */
BASE64_AVX512_TARGET
static inline int
base64_decode_64_avx512(const uint8_t *in, uint8_t *out)
{
    /* Load 64 input characters */
    __m512i input = _mm512_loadu_si512((const __m512i *)in);

    /* Check for padding '=' (61) - if present, exit to scalar path */
    __mmask64 pad_mask = _mm512_cmpeq_epi8_mask(input, _mm512_set1_epi8('='));
    if (pad_mask != 0) {
        return -1;
    }

    /* Check all characters are ASCII (< 128) */
    __mmask64 high_mask = _mm512_cmpgt_epi8_mask(input, _mm512_set1_epi8(127));
    if (high_mask != 0) {
        return -1;
    }

    /* Load decode table (need two 64-byte halves for 128-byte table) */
    __m512i table_lo = _mm512_load_si512((const __m512i *)&_b64_decode_table[0]);
    __m512i table_hi = _mm512_load_si512((const __m512i *)&_b64_decode_table[64]);

    /* Look up each character in the decode table */
    /* For chars 0-63, use table_lo; for chars 64-127, use table_hi */
    __mmask64 hi_select = _mm512_cmpge_epu8_mask(input, _mm512_set1_epi8(64));

    /* Adjust indices for high table lookup */
    __m512i idx_lo = input;
    __m512i idx_hi = _mm512_sub_epi8(input, _mm512_set1_epi8(64));

    /* Look up from both tables */
    __m512i val_lo = _mm512_permutexvar_epi8(idx_lo, table_lo);
    __m512i val_hi = _mm512_permutexvar_epi8(idx_hi, table_hi);

    /* Select correct result based on input range */
    __m512i values = _mm512_mask_blend_epi8(hi_select, val_lo, val_hi);

    /* Check for invalid characters (decode table returns -1 / 0xFF) */
    __mmask64 invalid_mask = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), values);
    if (invalid_mask != 0) {
        return -1;
    }

    /*
     * Now combine 6-bit values into bytes.
     * 4 sextets [a,b,c,d] (each 6 bits) -> 3 bytes:
     *   byte0 = (a << 2) | (b >> 4)
     *   byte1 = (b << 4) | (c >> 2)
     *   byte2 = (c << 6) | d
     *
     * Use multiply-add: vpmaddubsw and vpmaddwd
     */

    /* Pack using multiply-add trick */
    /* For each group of 4 bytes [a,b,c,d], compute:
     * word0 = a*64 + b = (a << 6) | b  (12 bits in 16-bit word)
     * word1 = c*64 + d = (c << 6) | d  (12 bits in 16-bit word)
     */
    __m512i merge1 = _mm512_maddubs_epi16(values, _mm512_set1_epi32(0x01400140));

    /* Now merge pairs of 16-bit words:
     * dword = word0 * 4096 + word1 = (word0 << 12) | word1  (24 bits in 32-bit dword)
     */
    __m512i merge2 = _mm512_madd_epi16(merge1, _mm512_set1_epi32(0x00011000));

    /*
     * Shuffle to extract the 3 valid bytes from each 32-bit word.
     * We need 48 output bytes from 64 input bytes (16 x 32-bit words).
     * Each word contributes 3 bytes (bytes 0,1,2 in little-endian), skip byte 3.
     *
     * Use _mm512_permutexvar_epi8 for cross-lane byte permutation.
     * Word n occupies bytes 4n to 4n+3, we want bytes 4n, 4n+1, 4n+2.
     */
    /*
     * After merge2, each 32-bit word has bytes laid out as:
     *   byte 0 = ((c << 6) | d) & 0xFF  -> goes to output byte 2
     *   byte 1 = ((b << 4) | (c >> 2))  -> goes to output byte 1
     *   byte 2 = ((a << 2) | (b >> 4))  -> goes to output byte 0
     *   byte 3 = 0 (unused)
     *
     * So we need to reverse the byte order: take [2,1,0] from each word.
     */
    const __m512i pack_shuf = _mm512_set_epi8(
        /* Bytes 63-48: padding (will be masked out) */
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        /* Bytes 47-36: words 12-15 (take [2,1,0] from each word) */
        60, 61, 62,  56, 57, 58,  52, 53, 54,  48, 49, 50,
        /* Bytes 35-24: words 8-11 */
        44, 45, 46,  40, 41, 42,  36, 37, 38,  32, 33, 34,
        /* Bytes 23-12: words 4-7 */
        28, 29, 30,  24, 25, 26,  20, 21, 22,  16, 17, 18,
        /* Bytes 11-0: words 0-3 */
        12, 13, 14,   8,  9, 10,   4,  5,  6,   0,  1,  2
    );

    /* Use vpermb for cross-lane byte shuffle */
    __m512i packed = _mm512_permutexvar_epi8(pack_shuf, merge2);

    /* Store 48 bytes (only lower 48 bytes are valid) */
    _mm512_mask_storeu_epi8(out, 0x0000FFFFFFFFFFFF, packed);

    return 48;
}


/*
 * Encode using AVX-512 VBMI
 * Returns number of input bytes processed (multiple of 48)
 */
BASE64_AVX512_TARGET
static Py_ssize_t
base64_encode_avx512vbmi(const unsigned char *in, Py_ssize_t in_len,
                         unsigned char *out, const unsigned char *table)
{
    Py_ssize_t blocks = in_len / 48;
    (void)table;

    for (Py_ssize_t i = 0; i < blocks; i++) {
        base64_encode_48_avx512(in + i * 48, out + i * 64);
    }

    return blocks * 48;
}

/*
 * Decode using AVX-512 VBMI
 * Returns number of input bytes processed (multiple of 64)
 */
BASE64_AVX512_TARGET
static Py_ssize_t
base64_decode_avx512vbmi(const unsigned char *in, Py_ssize_t in_len,
                         unsigned char *out, const unsigned char *table)
{
    Py_ssize_t blocks = in_len / 64;
    (void)table;

    for (Py_ssize_t i = 0; i < blocks; i++) {
        if (base64_decode_64_avx512(in + i * 64, out + i * 48) < 0) {
            return i * 64;  /* Stop at invalid/padding */
        }
    }

    return blocks * 64;
}

#endif /* BASE64_HAS_AVX512_COMPILED */

#endif /* BASE64_X86_64 */

#endif /* BASE64_SIMD_H */
