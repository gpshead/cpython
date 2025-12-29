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
 *
 * CPU features are detected once during module initialization via
 * _base64_init_cpu_features(), called from binascii_exec().
 * This avoids any potential data races from lazy initialization.
 */

/* Feature detection state: single value avoids atomicity concerns */
enum {
    _BASE64_CPU_UNKNOWN = 0,   /* Not yet checked */
    _BASE64_CPU_NO_AVX512 = 1, /* Checked, no AVX-512 VBMI */
    _BASE64_CPU_HAS_AVX512 = 2 /* Checked, has AVX-512 VBMI */
};

static int _base64_cpu_features = _BASE64_CPU_UNKNOWN;

#ifdef BASE64_X86_64

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

/*
 * Initialize CPU feature detection. Call once from module init.
 */
static void
_base64_init_cpu_features(void)
{
    if (_base64_cpu_features != _BASE64_CPU_UNKNOWN) {
        return;  /* Already initialized */
    }

    int has_avx512vbmi = 0;
    int eax, ebx, ecx, edx;
    CPUID(0, eax, ebx, ecx, edx);

    if (eax >= 7) {
        CPUID(7, eax, ebx, ecx, edx);
        /* AVX-512 VBMI is bit 1 of ECX (leaf 7) */
        has_avx512vbmi = (ecx >> 1) & 1;
    }

    _base64_cpu_features = has_avx512vbmi
        ? _BASE64_CPU_HAS_AVX512
        : _BASE64_CPU_NO_AVX512;
}

static inline int
base64_has_avx512vbmi(void)
{
    return _base64_cpu_features == _BASE64_CPU_HAS_AVX512;
}

#else

static void _base64_init_cpu_features(void) { }
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

/*
 * The base64 encoding table as a 512-bit vector.
 * Not null-terminated (exactly 64 chars), hence the nonstring attribute
 * to suppress -Wunterminated-string-initialization on GCC 15+.
 */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((nonstring))
#endif
static const char Py_ALIGNED(64) _b64_table[64] =
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
     * Reshuffle 48 input bytes (16 triplets) into 64 bytes for 6-bit extraction.
     *
     * Formula: For triplet n (bytes b0=in[3n], b1=in[3n+1], b2=in[3n+2]):
     *   out[4n:4n+3] = [b1, b0, b2, b1]
     *
     * This layout enables the multiply-mask trick for extracting 4 sextets
     * from each 32-bit word. _mm512_set_epi8 args go high-to-low (63 to 0).
     */
    const __m512i shuf_input = _mm512_set_epi8(
        /* Triplet 15 [45,46,47] */ 46, 47, 45, 46,
        /* Triplet 14 [42,43,44] */ 43, 44, 42, 43,
        /* Triplet 13 [39,40,41] */ 40, 41, 39, 40,
        /* Triplet 12 [36,37,38] */ 37, 38, 36, 37,
        /* Triplet 11 [33,34,35] */ 34, 35, 33, 34,
        /* Triplet 10 [30,31,32] */ 31, 32, 30, 31,
        /* Triplet  9 [27,28,29] */ 28, 29, 27, 28,
        /* Triplet  8 [24,25,26] */ 25, 26, 24, 25,
        /* Triplet  7 [21,22,23] */ 22, 23, 21, 22,
        /* Triplet  6 [18,19,20] */ 19, 20, 18, 19,
        /* Triplet  5 [15,16,17] */ 16, 17, 15, 16,
        /* Triplet  4 [12,13,14] */ 13, 14, 12, 13,
        /* Triplet  3  [9,10,11] */ 10, 11,  9, 10,
        /* Triplet  2   [6,7,8]  */  7,  8,  6,  7,
        /* Triplet  1   [3,4,5]  */  4,  5,  3,  4,
        /* Triplet  0   [0,1,2]  */  1,  2,  0,  1
    );

    /* Use vpermb for cross-lane byte shuffle */
    __m512i triplets = _mm512_permutexvar_epi8(shuf_input, input);

    /*
     * Extract 4 sextets from each 32-bit word using the multiply-mask trick.
     * After shuffle, each word = (b0 << 16) | (b1 << 8) | b2 (little-endian).
     * We extract bits at positions [18:23], [12:17], [6:11], [0:5].
     */
    __m512i t0 = _mm512_and_si512(triplets, _mm512_set1_epi32(0x0FC0FC00));
    __m512i t1 = _mm512_and_si512(triplets, _mm512_set1_epi32(0x003F03F0));
    t0 = _mm512_mulhi_epu16(t0, _mm512_set1_epi32(0x04000040));
    t1 = _mm512_mullo_epi16(t1, _mm512_set1_epi32(0x01000010));
    __m512i indices = _mm512_or_si512(t0, t1);

    /* Use vpermb to look up all 64 characters at once from the 64-byte table */
    __m512i result = _mm512_permutexvar_epi8(indices, table);

    /* Store 64 output characters */
    _mm512_storeu_si512((__m512i *)out, result);
}


/*
 * Build the decode lookup table.
 * Maps ASCII character to 6-bit value (0-63), or 0xFF for invalid.
 * Only first 128 entries needed (ASCII).
 */
static const int8_t Py_ALIGNED(64) _b64_decode_table[128] = {
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
     * Pack 16 x 32-bit words into 48 output bytes.
     *
     * Formula: For word n at bytes [4n:4n+3], extract bytes [4n+2, 4n+1, 4n]
     * (reverse order, skip byte 3). Output goes to bytes [3n:3n+2].
     * Upper 16 bytes are padding (masked out on store).
     */
    const __m512i pack_shuf = _mm512_set_epi8(
        /* Padding (bytes 63-48, masked out) */
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        /* Words 15-12 */ 62, 61, 60,  58, 57, 56,  54, 53, 52,  50, 49, 48,
        /* Words 11-8  */ 46, 45, 44,  42, 41, 40,  38, 37, 36,  34, 33, 32,
        /* Words  7-4  */ 30, 29, 28,  26, 25, 24,  22, 21, 20,  18, 17, 16,
        /* Words  3-0  */ 14, 13, 12,  10,  9,  8,   6,  5,  4,   2,  1,  0
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
                         unsigned char *out,
                         const unsigned char *Py_UNUSED(table))
{
    Py_ssize_t blocks = in_len / 48;

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
                         unsigned char *out,
                         const unsigned char *Py_UNUSED(table))
{
    Py_ssize_t blocks = in_len / 64;

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
 * Uses arithmetic encoding (no table lookups) since NEON's tbl instruction
 * is limited to 16-32 byte tables, insufficient for base64's 64-char alphabet.
 */
static inline void
base64_encode_12_neon(const uint8_t *in, uint8_t *out)
{
    /* Load 12 input bytes (loads 16, only first 12 used) */
    uint8x16_t input = vld1q_u8(in);

    /*
     * Reshuffle 4 triplets for 16-bit pair extraction.
     * Formula: For triplet n (bytes b0=in[3n], b1=in[3n+1], b2=in[3n+2]):
     *   out[4n:4n+3] = [b1, b0, b2, b1]
     * This creates 16-bit pairs: even=[b1|b0<<8], odd=[b2|b1<<8]
     */
    static const uint8_t shuf_tbl[16] = {
         1,  0,  2,  1,    /* triplet 0 */
         4,  3,  5,  4,    /* triplet 1 */
         7,  6,  8,  7,    /* triplet 2 */
        10,  9, 11, 10     /* triplet 3 */
    };
    uint8x16_t shuf = vld1q_u8(shuf_tbl);
    uint8x16_t reshuffled = vqtbl1q_u8(input, shuf);

    /*
     * Extract 4 sextets from each triplet using 16-bit arithmetic.
     * From even pair (b1 | b0<<8): s0 = bits[10:15], s1 = bits[4:9]
     * From odd pair  (b2 | b1<<8): s2 = bits[6:11],  s3 = bits[0:5]
     */
    uint16x8_t in16 = vreinterpretq_u16_u8(reshuffled);
    uint16x4_t even = vuzp1_u16(vget_low_u16(in16), vget_high_u16(in16));
    uint16x4_t odd = vuzp2_u16(vget_low_u16(in16), vget_high_u16(in16));

    uint16x4_t s0 = vshr_n_u16(even, 10);
    uint16x4_t s1 = vand_u16(vshr_n_u16(even, 4), vdup_n_u16(0x3F));
    uint16x4_t s2 = vand_u16(vshr_n_u16(odd, 6), vdup_n_u16(0x3F));
    uint16x4_t s3 = vand_u16(odd, vdup_n_u16(0x3F));

    /* Interleave sextets back to [s0,s1,s2,s3] order per triplet */
    uint16x4_t s01_lo = vzip1_u16(s0, s1);
    uint16x4_t s01_hi = vzip2_u16(s0, s1);
    uint16x4_t s23_lo = vzip1_u16(s2, s3);
    uint16x4_t s23_hi = vzip2_u16(s2, s3);
    uint16x8_t indices_lo = vcombine_u16(s01_lo, s23_lo);
    uint16x8_t indices_hi = vcombine_u16(s01_hi, s23_hi);

    /* Narrow to 8-bit and reorder to final positions */
    uint8x8_t idx_lo = vmovn_u16(indices_lo);
    uint8x8_t idx_hi = vmovn_u16(indices_hi);
    static const uint8_t reorder_tbl[16] = {
        0, 1, 4, 5, 2, 3, 6, 7,
        8, 9, 12, 13, 10, 11, 14, 15
    };
    uint8x16_t indices = vcombine_u8(idx_lo, idx_hi);
    uint8x16_t reorder = vld1q_u8(reorder_tbl);
    indices = vqtbl1q_u8(indices, reorder);

    /*
     * Convert 6-bit indices to ASCII using arithmetic offset adjustment.
     * Base offset is 'A' (65), adjusted per range:
     *   0-25:  +65 -> 'A'-'Z'
     *   26-51: +71 -> 'a'-'z'
     *   52-61: -4  -> '0'-'9'
     *   62:    -19 -> '+'
     *   63:    -16 -> '/'
     */
    uint8x16_t offset = vdupq_n_u8(65);
    uint8x16_t ge26 = vcgeq_u8(indices, vdupq_n_u8(26));
    offset = vaddq_u8(offset, vandq_u8(ge26, vdupq_n_u8(6)));
    uint8x16_t ge52 = vcgeq_u8(indices, vdupq_n_u8(52));
    offset = vsubq_u8(offset, vandq_u8(ge52, vdupq_n_u8(75)));
    uint8x16_t eq62 = vceqq_u8(indices, vdupq_n_u8(62));
    offset = vsubq_u8(offset, vandq_u8(eq62, vdupq_n_u8(15)));
    uint8x16_t eq63 = vceqq_u8(indices, vdupq_n_u8(63));
    offset = vsubq_u8(offset, vandq_u8(eq63, vdupq_n_u8(12)));

    uint8x16_t result = vaddq_u8(indices, offset);
    vst1q_u8(out, result);
}


/*
 * Decode 16 base64 characters to 12 bytes using NEON
 * Returns 12 on success, -1 if padding or invalid character found
 */
static inline int
base64_decode_16_neon(const uint8_t *in, uint8_t *out)
{
    uint8x16_t input = vld1q_u8(in);

    /* Check for padding '=' - bail to scalar if found */
    uint8x16_t eq_mask = vceqq_u8(input, vdupq_n_u8('='));
    if (vmaxvq_u8(eq_mask) != 0) {
        return -1;
    }

    /*
     * Convert ASCII to 6-bit values using arithmetic.
     * Invalid chars produce 0xFF which fails the >=64 check.
     */
    uint8x16_t values = vdupq_n_u8(0xFF);

    /* 'A'-'Z' -> 0-25 */
    uint8x16_t is_upper = vandq_u8(vcgeq_u8(input, vdupq_n_u8('A')),
                                   vcleq_u8(input, vdupq_n_u8('Z')));
    values = vbslq_u8(is_upper, vsubq_u8(input, vdupq_n_u8('A')), values);

    /* 'a'-'z' -> 26-51 */
    uint8x16_t is_lower = vandq_u8(vcgeq_u8(input, vdupq_n_u8('a')),
                                   vcleq_u8(input, vdupq_n_u8('z')));
    values = vbslq_u8(is_lower, vsubq_u8(input, vdupq_n_u8('a' - 26)), values);

    /* '0'-'9' -> 52-61 */
    uint8x16_t is_digit = vandq_u8(vcgeq_u8(input, vdupq_n_u8('0')),
                                   vcleq_u8(input, vdupq_n_u8('9')));
    values = vbslq_u8(is_digit, vaddq_u8(input, vdupq_n_u8(4)), values);

    /* '+' -> 62, '/' -> 63 */
    values = vbslq_u8(vceqq_u8(input, vdupq_n_u8('+')), vdupq_n_u8(62), values);
    values = vbslq_u8(vceqq_u8(input, vdupq_n_u8('/')), vdupq_n_u8(63), values);

    /* Check for invalid (any value >= 64) */
    if (vmaxvq_u8(vcgeq_u8(values, vdupq_n_u8(64))) != 0) {
        return -1;
    }

    /*
     * Merge sextets into bytes using multiply-add.
     * Step 1: Pair sextets -> 12-bit values: m[i] = s[2i]*64 + s[2i+1]
     * Step 2: Pair 12-bit values -> 24-bit: out = (m_even << 12) | m_odd
     */
    uint8x8x2_t deinterleaved = vuzp_u8(vget_low_u8(values), vget_high_u8(values));
    uint8x8_t evens = deinterleaved.val[0];
    uint8x8_t odds = deinterleaved.val[1];
    uint16x8_t merged = vmlal_u8(vmovl_u8(odds), evens, vdup_n_u8(64));

    uint16x4_t m_even = vuzp1_u16(vget_low_u16(merged), vget_high_u16(merged));
    uint16x4_t m_odd = vuzp2_u16(vget_low_u16(merged), vget_high_u16(merged));
    uint32x4_t combined = vorrq_u32(
        vshlq_n_u32(vmovl_u16(m_even), 12),
        vmovl_u16(m_odd)
    );

    /*
     * Reorder bytes: each 32-bit word has [b2,b1,b0,0] in little-endian.
     * Extract [b0,b1,b2] from each word (bytes at offsets 2,1,0).
     */
    uint8x16_t bytes = vreinterpretq_u8_u32(combined);
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
                   unsigned char *out, const unsigned char *Py_UNUSED(table))
{
    Py_ssize_t blocks = in_len / 12;

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
                   unsigned char *out, const unsigned char *Py_UNUSED(table))
{
    Py_ssize_t blocks = in_len / 16;

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
