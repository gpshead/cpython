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

#if defined(__aarch64__) || defined(_M_ARM64)
#define BASE64_ARM64
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

/* The base64 encoding table as a 512-bit vector (not null-terminated) */
static const char _b64_table[64]
    __attribute__((aligned(64), nonstring)) =
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


/*
 * ARM NEON Implementation
 *
 * Uses arithmetic operations instead of table lookups since NEON's tbl
 * instruction is limited to 16-32 byte tables.
 *
 * Encoding: 12 input bytes -> 16 output chars per iteration
 * Decoding: 16 input chars -> 12 output bytes per iteration
 */
#ifdef BASE64_ARM64

#include <arm_neon.h>

#define BASE64_HAS_NEON 1

/*
 * Encode 12 bytes to 16 base64 characters using NEON
 *
 * The arithmetic encoding avoids table lookups by computing the ASCII
 * values directly from the 6-bit indices:
 *   0-25  -> 'A'-'Z' (add 'A' = 65)
 *   26-51 -> 'a'-'z' (add 'a'-26 = 71)
 *   52-61 -> '0'-'9' (subtract 4, since '0'=48 and 52-4=48)
 *   62    -> '+'     (subtract 19, since 62-19=43='+')
 *   63    -> '/'     (subtract 16, since 63-16=47='/')
 */
static inline void
base64_encode_12_neon(const uint8_t *in, uint8_t *out)
{
    /*
     * Load 12 input bytes into a 16-byte vector.
     * We load 16 bytes but only use the first 12.
     * Layout: [b0,b1,b2, b3,b4,b5, b6,b7,b8, b9,b10,b11, x,x,x,x]
     */
    uint8x16_t input = vld1q_u8(in);

    /*
     * Reshuffle bytes for 6-bit extraction.
     * For each triplet (b0,b1,b2), we need to extract 4 sextets.
     * Arrange as: [b0,b0,b1,b2] for each triplet to enable the
     * shift-and-mask extraction.
     *
     * Input triplets at indices: [0,1,2], [3,4,5], [6,7,8], [9,10,11]
     * Output 16 bytes, 4 per triplet.
     */
    static const uint8_t shuf_tbl[16] = {
         0,  0,  1,  2,    /*  triplet 0: b0,b0,b1,b2 */
         3,  3,  4,  5,    /*  triplet 1 */
         6,  6,  7,  8,    /*  triplet 2 */
         9,  9, 10, 11     /*  triplet 3 */
    };
    uint8x16_t shuf = vld1q_u8(shuf_tbl);
    uint8x16_t shuffled = vqtbl1q_u8(input, shuf);

    /*
     * Now each group of 4 bytes contains [b0, b0, b1, b2].
     * Reinterpret as 32-bit words for extraction.
     *
     * For each 32-bit word (little-endian): b2 | (b1<<8) | (b0<<16) | (b0<<24)
     *
     * Extract sextets using shifts and masks:
     *   sextet0 = (b0 >> 2) & 0x3F
     *   sextet1 = ((b0 << 4) | (b1 >> 4)) & 0x3F
     *   sextet2 = ((b1 << 2) | (b2 >> 6)) & 0x3F
     *   sextet3 = b2 & 0x3F
     */

    /* Split into even/odd 16-bit pairs for the multiply-mask trick */
    uint32x4_t in32 = vreinterpretq_u32_u8(shuffled);

    /* Mask and shift to extract sextets */
    /* After shuffle, as 32-bit LE: word = b2 | (b1<<8) | (b0<<16) | (b0<<24) */

    /* Extract using AND + shift approach */
    uint32x4_t mask0 = vdupq_n_u32(0x0000003F);  /* sextet3: bits 0-5 */
    uint32x4_t mask1 = vdupq_n_u32(0x00003F00);  /* sextet2: bits 8-13 */
    uint32x4_t mask2 = vdupq_n_u32(0x003F0000);  /* sextet1: bits 16-21 */
    uint32x4_t mask3 = vdupq_n_u32(0x3F000000);  /* sextet0: bits 24-29 */

    /* But we need different bit positions... Let me recalculate. */
    /*
     * After vqtbl1q_u8 with our shuffle, byte layout is:
     * [b0, b0, b1, b2, b0, b0, b1, b2, ...]
     *
     * As 32-bit little-endian:
     * word = b2 | (b1 << 8) | (b0 << 16) | (b0 << 24)
     *
     * We want:
     *   out[0] = (b0 >> 2) & 0x3F         = bits 18-23 of word, shifted
     *   out[1] = ((b0&3)<<4) | (b1>>4)    = bits 12-17
     *   out[2] = ((b1&0xF)<<2) | (b2>>6)  = bits 6-11
     *   out[3] = b2 & 0x3F                = bits 0-5
     *
     * Use a different approach: work with 16-bit lanes.
     */

    /*
     * Alternative: use the standard reshuffle pattern [b1, b0, b2, b1]
     * which works better with the multiply-mask trick.
     */
    static const uint8_t shuf2_tbl[16] = {
         1,  0,  2,  1,    /* triplet 0: b1,b0,b2,b1 */
         4,  3,  5,  4,    /* triplet 1 */
         7,  6,  8,  7,    /* triplet 2 */
        10,  9, 11, 10     /* triplet 3 */
    };
    uint8x16_t shuf2 = vld1q_u8(shuf2_tbl);
    uint8x16_t reshuffled = vqtbl1q_u8(input, shuf2);

    /*
     * Now as 32-bit LE: word = b1 | (b0 << 8) | (b2 << 16) | (b1 << 24)
     * Wait, that's not right either. Let me be more careful.
     *
     * Bytes in memory after shuffle: [b1, b0, b2, b1, ...]
     * As 32-bit little-endian load:
     *   byte[0] = b1 -> bits 0-7
     *   byte[1] = b0 -> bits 8-15
     *   byte[2] = b2 -> bits 16-23
     *   byte[3] = b1 -> bits 24-31
     *
     * We need to extract:
     *   sextet0 = b0[7:2] = bits 10-15 of word >> 10, masked
     *   sextet1 = b0[1:0]:b1[7:4] = ((word >> 4) & 0x3F) for low bits...
     *
     * Actually, let's use the cleaner multiply approach:
     */

    /* Use the proven multiply-and-mask technique */
    uint16x8_t in16 = vreinterpretq_u16_u8(reshuffled);

    /*
     * For 16-bit pairs, with layout [b1,b0] and [b2,b1] alternating:
     * Pair 0 (bytes 0-1): b1 | (b0 << 8) as 16-bit LE
     * Pair 1 (bytes 2-3): b2 | (b1 << 8) as 16-bit LE
     *
     * We want sextets from 24 bits [b0, b1, b2]:
     *   s0 = b0 >> 2
     *   s1 = (b0 << 4 | b1 >> 4) & 0x3F
     *   s2 = (b1 << 2 | b2 >> 6) & 0x3F
     *   s3 = b2 & 0x3F
     *
     * From pair0 = b1 | (b0 << 8):
     *   s0 = (pair0 >> 10) & 0x3F
     *   s1 = (pair0 >> 4) & 0x3F
     *
     * From pair1 = b2 | (b1 << 8):
     *   s2 = (pair1 >> 6) & 0x3F
     *   s3 = pair1 & 0x3F
     */

    /* Split into even and odd 16-bit elements */
    uint16x4_t even = vuzp1_u16(vget_low_u16(in16), vget_high_u16(in16));
    uint16x4_t odd = vuzp2_u16(vget_low_u16(in16), vget_high_u16(in16));

    /* even contains [pair0_t0, pair0_t1, pair0_t2, pair0_t3] = [b1|b0<<8, ...] */
    /* odd contains  [pair1_t0, pair1_t1, pair1_t2, pair1_t3] = [b2|b1<<8, ...] */

    /* Extract sextets */
    uint16x4_t s0 = vshr_n_u16(even, 10);
    uint16x4_t s1 = vand_u16(vshr_n_u16(even, 4), vdup_n_u16(0x3F));
    uint16x4_t s2 = vand_u16(vshr_n_u16(odd, 6), vdup_n_u16(0x3F));
    uint16x4_t s3 = vand_u16(odd, vdup_n_u16(0x3F));

    /* Interleave back: we want [s0,s1,s2,s3] for each triplet */
    /* Zip s0,s1 and s2,s3, then zip those results */
    uint16x4_t s01_lo = vzip1_u16(s0, s1);  /* [s0_0, s1_0, s0_1, s1_1] */
    uint16x4_t s01_hi = vzip2_u16(s0, s1);  /* [s0_2, s1_2, s0_3, s1_3] */
    uint16x4_t s23_lo = vzip1_u16(s2, s3);
    uint16x4_t s23_hi = vzip2_u16(s2, s3);

    /* Combine into final order */
    uint16x8_t indices_lo = vcombine_u16(s01_lo, s23_lo);
    uint16x8_t indices_hi = vcombine_u16(s01_hi, s23_hi);

    /* We have 16-bit indices but need 8-bit. Narrow them. */
    uint8x8_t idx_lo = vmovn_u16(indices_lo);
    uint8x8_t idx_hi = vmovn_u16(indices_hi);

    /* Reorder: currently [s0_0,s1_0,s0_1,s1_1,s2_0,s3_0,s2_1,s3_1] in idx_lo */
    /* We want [s0_0,s1_0,s2_0,s3_0,s0_1,s1_1,s2_1,s3_1] */
    static const uint8_t reorder_tbl[16] = {
        0, 1, 4, 5, 2, 3, 6, 7,   /* reorder first 8 */
        8, 9, 12, 13, 10, 11, 14, 15  /* reorder second 8 */
    };
    uint8x16_t indices = vcombine_u8(idx_lo, idx_hi);
    uint8x16_t reorder = vld1q_u8(reorder_tbl);
    indices = vqtbl1q_u8(indices, reorder);

    /*
     * Now convert 6-bit indices to ASCII using arithmetic.
     *
     * idx < 26:  add 'A' (65)
     * idx < 52:  add 'a' - 26 = 71
     * idx < 62:  add '0' - 52 = -4 (or subtract 4)
     * idx == 62: result is '+' (43), so add 43 - 62 = -19
     * idx == 63: result is '/' (47), so add 47 - 63 = -16
     *
     * We can compute an offset for each index and add it.
     */
    uint8x16_t offset = vdupq_n_u8(65);  /* Start with 'A' offset */

    /* Adjust for indices >= 26: add (71 - 65) = 6 */
    uint8x16_t ge26 = vcgeq_u8(indices, vdupq_n_u8(26));
    offset = vaddq_u8(offset, vandq_u8(ge26, vdupq_n_u8(6)));

    /* Adjust for indices >= 52: add (-4 - 71) = -75, effectively (256-75)=181 */
    uint8x16_t ge52 = vcgeq_u8(indices, vdupq_n_u8(52));
    offset = vsubq_u8(offset, vandq_u8(ge52, vdupq_n_u8(75)));

    /* Adjust for index 62: add (-19 - (-4)) = -15 */
    uint8x16_t eq62 = vceqq_u8(indices, vdupq_n_u8(62));
    offset = vsubq_u8(offset, vandq_u8(eq62, vdupq_n_u8(15)));

    /* Adjust for index 63: add (-16 - (-4)) = -12 */
    uint8x16_t eq63 = vceqq_u8(indices, vdupq_n_u8(63));
    offset = vsubq_u8(offset, vandq_u8(eq63, vdupq_n_u8(12)));

    /* Apply offset to get ASCII */
    uint8x16_t result = vaddq_u8(indices, offset);

    /* Store 16 output characters */
    vst1q_u8(out, result);
}


/*
 * Decode 16 base64 characters to 12 bytes using NEON
 * Returns 12 on success, -1 if invalid character found
 */
static inline int
base64_decode_16_neon(const uint8_t *in, uint8_t *out)
{
    /* Load 16 input characters */
    uint8x16_t input = vld1q_u8(in);

    /* Check for padding '=' - bail to scalar if found */
    uint8x16_t eq_mask = vceqq_u8(input, vdupq_n_u8('='));
    if (vmaxvq_u8(eq_mask) != 0) {
        return -1;
    }

    /*
     * Convert ASCII to 6-bit values using arithmetic (reverse of encode).
     *
     * 'A'-'Z' (65-90)  -> 0-25:  subtract 65
     * 'a'-'z' (97-122) -> 26-51: subtract 71
     * '0'-'9' (48-57)  -> 52-61: add 4
     * '+' (43)         -> 62:    add 19
     * '/' (47)         -> 63:    add 16
     *
     * Invalid chars should produce values >= 64.
     */
    uint8x16_t values = vdupq_n_u8(0xFF);  /* Start invalid */

    /* Handle 'A'-'Z': 65-90 -> 0-25 */
    uint8x16_t is_upper = vandq_u8(vcgeq_u8(input, vdupq_n_u8('A')),
                                   vcleq_u8(input, vdupq_n_u8('Z')));
    values = vbslq_u8(is_upper, vsubq_u8(input, vdupq_n_u8('A')), values);

    /* Handle 'a'-'z': 97-122 -> 26-51 */
    uint8x16_t is_lower = vandq_u8(vcgeq_u8(input, vdupq_n_u8('a')),
                                   vcleq_u8(input, vdupq_n_u8('z')));
    values = vbslq_u8(is_lower, vsubq_u8(input, vdupq_n_u8('a' - 26)), values);

    /* Handle '0'-'9': 48-57 -> 52-61 */
    uint8x16_t is_digit = vandq_u8(vcgeq_u8(input, vdupq_n_u8('0')),
                                   vcleq_u8(input, vdupq_n_u8('9')));
    values = vbslq_u8(is_digit, vaddq_u8(input, vdupq_n_u8(4)), values);

    /* Handle '+': 43 -> 62 */
    uint8x16_t is_plus = vceqq_u8(input, vdupq_n_u8('+'));
    values = vbslq_u8(is_plus, vdupq_n_u8(62), values);

    /* Handle '/': 47 -> 63 */
    uint8x16_t is_slash = vceqq_u8(input, vdupq_n_u8('/'));
    values = vbslq_u8(is_slash, vdupq_n_u8(63), values);

    /* Check for invalid (any value >= 64 means invalid input) */
    uint8x16_t invalid = vcgeq_u8(values, vdupq_n_u8(64));
    if (vmaxvq_u8(invalid) != 0) {
        return -1;
    }

    /*
     * Combine four 6-bit values into three bytes.
     * [s0, s1, s2, s3] -> [b0, b1, b2] where:
     *   b0 = (s0 << 2) | (s1 >> 4)
     *   b1 = (s1 << 4) | (s2 >> 2)
     *   b2 = (s2 << 6) | s3
     *
     * Use multiply-add: vpmaddubsw equivalent on NEON.
     */

    /*
     * First merge pairs: s0,s1 -> 12-bit value, s2,s3 -> 12-bit value
     * Then merge those 12-bit pairs into 24-bit values.
     */

    /* Reinterpret as 16-bit for pair operations */
    uint16x8_t vals16 = vreinterpretq_u16_u8(values);

    /*
     * For each pair of bytes [s_lo, s_hi] in 16-bit (little-endian: s_hi<<8 | s_lo):
     * We want: (s_lo << 6) | s_hi = s_lo * 64 + s_hi
     *
     * Bytes are [s0,s1,s2,s3,s4,s5,...] = quads [s0,s1,s2,s3], [s4,s5,s6,s7],...
     * As 16-bit: [s1<<8|s0, s3<<8|s2, s5<<8|s4, s7<<8|s6, ...]
     *
     * We want pairs (s0,s1) and (s2,s3) etc.
     * First rearrange: swap adjacent bytes to get [s0<<8|s1, s2<<8|s3, ...]
     */
    uint8x16_t swapped = vrev16q_u8(values);  /* Swap bytes within 16-bit lanes */
    uint16x8_t swapped16 = vreinterpretq_u16_u8(swapped);

    /*
     * Now swapped16[0] = s0<<8 | s1, swapped16[1] = s2<<8 | s3, etc.
     *
     * Merge: (s0 << 6) | s1 = ((s0<<8|s1) >> 2) & 0x0FFF
     *        Actually: s0*64 + s1
     * Since s0 is in high byte: ((s0<<8) >> 2) | s1 = (s0 << 6) | s1
     * That's: (swapped16 >> 2) for high part, but we need to handle low byte.
     *
     * Let me use a clearer approach with explicit multiply-add.
     */

    /* Extract odd/even bytes for multiply-add */
    uint8x8x2_t deinterleaved = vuzp_u8(vget_low_u8(values), vget_high_u8(values));
    uint8x8_t evens = deinterleaved.val[0];  /* s0, s2, s4, s6, s8, s10, s12, s14 */
    uint8x8_t odds = deinterleaved.val[1];   /* s1, s3, s5, s7, s9, s11, s13, s15 */

    /* Multiply evens by 64 and add odds: result = even*64 + odd */
    uint16x8_t merged = vmlal_u8(vmovl_u8(odds), evens, vdup_n_u8(64));
    /* merged[i] = s_{2i} * 64 + s_{2i+1}, which is 12 bits */

    /*
     * Now merge pairs of 12-bit values into 24-bit values.
     * merged = [m0, m1, m2, m3, m4, m5, m6, m7] where each m is 12 bits
     *
     * We want: (m0 << 12) | m1, (m2 << 12) | m3, etc.
     *
     * Extract even and odd 16-bit lanes:
     */
    uint16x4_t m_even = vuzp1_u16(vget_low_u16(merged), vget_high_u16(merged));
    uint16x4_t m_odd = vuzp2_u16(vget_low_u16(merged), vget_high_u16(merged));

    /* m_even = [m0, m2, m4, m6], m_odd = [m1, m3, m5, m7] */

    /* Compute (m_even << 12) | m_odd as 32-bit */
    uint32x4_t combined = vorrq_u32(
        vshlq_n_u32(vmovl_u16(m_even), 12),
        vmovl_u16(m_odd)
    );

    /*
     * Now each 32-bit element contains 24 bits of output data.
     * combined[i] has bits 0-23 = output bytes [b0, b1, b2] for triplet i.
     *
     * In little-endian:
     *   byte 0 = bits 0-7   = b2
     *   byte 1 = bits 8-15  = b1
     *   byte 2 = bits 16-23 = b0
     *   byte 3 = 0
     *
     * We need to reverse bytes within each triplet and pack.
     */
    uint8x16_t bytes = vreinterpretq_u8_u32(combined);

    /*
     * Current layout: [b2,b1,b0,0, b2,b1,b0,0, b2,b1,b0,0, b2,b1,b0,0]
     * We want:        [b0,b1,b2, b0,b1,b2, b0,b1,b2, b0,b1,b2]
     *
     * Use tbl to reorder and pack.
     */
    static const uint8_t pack_tbl[16] = {
        2, 1, 0,    /* triplet 0 */
        6, 5, 4,    /* triplet 1 */
        10, 9, 8,   /* triplet 2 */
        14, 13, 12, /* triplet 3 */
        0xFF, 0xFF, 0xFF, 0xFF  /* padding (won't be stored) */
    };
    uint8x16_t pack_idx = vld1q_u8(pack_tbl);
    uint8x16_t packed = vqtbl1q_u8(bytes, pack_idx);

    /* Store 12 output bytes */
    vst1_u8(out, vget_low_u8(packed));
    vst1_lane_u32((uint32_t *)(out + 8), vreinterpret_u32_u8(vget_high_u8(packed)), 0);

    return 12;
}


/*
 * Encode using NEON
 * Returns number of input bytes processed (multiple of 12)
 */
static Py_ssize_t
base64_encode_neon(const unsigned char *in, Py_ssize_t in_len,
                   unsigned char *out, const unsigned char *table)
{
    Py_ssize_t blocks = in_len / 12;
    (void)table;

    for (Py_ssize_t i = 0; i < blocks; i++) {
        base64_encode_12_neon(in + i * 12, out + i * 16);
    }

    return blocks * 12;
}


/*
 * Decode using NEON
 * Returns number of input bytes processed (multiple of 16)
 */
static Py_ssize_t
base64_decode_neon(const unsigned char *in, Py_ssize_t in_len,
                   unsigned char *out, const unsigned char *table)
{
    Py_ssize_t blocks = in_len / 16;
    (void)table;

    for (Py_ssize_t i = 0; i < blocks; i++) {
        if (base64_decode_16_neon(in + i * 16, out + i * 12) < 0) {
            return i * 16;  /* Stop at invalid/padding */
        }
    }

    return blocks * 16;
}

#else

#define BASE64_HAS_NEON 0

#endif /* BASE64_ARM64 */


#endif /* BASE64_SIMD_H */
