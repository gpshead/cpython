/* Ryū: fast float-to-string conversion.
 *
 * This is an adaptation of Ulf Adams' Ryū algorithm and reference
 * implementation for use in CPython's short float repr.  See:
 *
 *     Ulf Adams. 2018. Ryū: Fast Float-to-String Conversion.
 *     In Proceedings of the 39th ACM SIGPLAN Conference on Programming
 *     Language Design and Implementation (PLDI '18).
 *     https://doi.org/10.1145/3192366.3192369
 *
 * Derived from https://github.com/ulfjack/ryu at commit
 * 4c0618b0e44f7ef027ebae05d2cc7812048f7c8f (files ryu/d2s.c,
 * ryu/common.h, ryu/d2s_intrinsics.h).
 *
 * The exported entry point _Py_ryu_dtoa() produces a result compatible
 * with _Py_dg_dtoa() in mode 0 (shortest representation): a digit
 * string, decimal-point position and sign.  double_repr_buffered() in
 * Python/pystrtod.c applies Python's repr formatting to this result.
 *
 * Major modifications from the upstream code:
 *
 *  0. Only the shortest-representation double path (d2d, d2d_small_int,
 *     write_digits) is kept; the upstream string formatter to_chars()
 *     and the float32 / fixed-precision code are not used.
 *
 *  1. The reference d2s_buffered_n() is replaced by _Py_ryu_dtoa(),
 *     which writes a digit-only string into a caller-provided buffer
 *     and returns the decimal-point position and sign separately,
 *     matching _Py_dg_dtoa() mode 0.
 *
 *  2. div5/div10/div100/div1e8 are written as explicit reciprocal
 *     multiplies via umulh() on all platforms, not just 32-bit ones,
 *     so that profile-guided optimisation cannot regress them to a
 *     hardware divide instruction.
 *
 *  3. RYU_DEBUG, RYU_OPTIMIZE_SIZE and RYU_32_BIT_PLATFORM conditionals
 *     are removed; HAVE___UINT128_T from pyconfig.h selects the
 *     128-bit multiply path.
 *
 * The body of d2d(), d2d_small_int(), the multiply helpers and the
 * digit-emission loop are kept in the upstream 2-space indentation
 * and brace style to make alignment with the reference code easy.
 *
 * ----------------------------------------------------------------------
 *
 * Copyright 2018 Ulf Adams
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Python.h"
#include "pycore_ryu.h"

#if _PY_SHORT_FLOAT_REPR == 1

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Power-of-5 lookup tables and the two-digit DIGIT_TABLE. */
#include "ryu_tables.h"

#define DOUBLE_MANTISSA_BITS 52
#define DOUBLE_EXPONENT_BITS 11
#define DOUBLE_BIAS 1023

/* ---------------------------------------------------------------------------
   Platform abstraction for 64x64->128 multiplication.
   --------------------------------------------------------------------------- */

#if defined(HAVE___UINT128_T) && !defined(_MSC_VER)
#  define RYU_HAS_UINT128 1
typedef __uint128_t ryu_uint128_t;
#elif defined(_MSC_VER) && defined(_M_X64)
#  define RYU_HAS_64_BIT_INTRINSICS 1
#  include <intrin.h>
#endif

#if !defined(RYU_HAS_UINT128)

static inline uint64_t umul128(const uint64_t a, const uint64_t b, uint64_t* const productHi) {
#if defined(RYU_HAS_64_BIT_INTRINSICS)
  return _umul128(a, b, productHi);
#else
  const uint32_t aLo = (uint32_t)a;
  const uint32_t aHi = (uint32_t)(a >> 32);
  const uint32_t bLo = (uint32_t)b;
  const uint32_t bHi = (uint32_t)(b >> 32);

  const uint64_t b00 = (uint64_t)aLo * bLo;
  const uint64_t b01 = (uint64_t)aLo * bHi;
  const uint64_t b10 = (uint64_t)aHi * bLo;
  const uint64_t b11 = (uint64_t)aHi * bHi;

  const uint32_t b00Hi = (uint32_t)(b00 >> 32);

  const uint64_t mid1 = b10 + b00Hi;
  const uint32_t mid1Lo = (uint32_t)(mid1);
  const uint32_t mid1Hi = (uint32_t)(mid1 >> 32);

  const uint64_t mid2 = b01 + mid1Lo;
  const uint32_t mid2Lo = (uint32_t)(mid2);
  const uint32_t mid2Hi = (uint32_t)(mid2 >> 32);

  const uint64_t pHi = b11 + mid1Hi + mid2Hi;
  const uint64_t pLo = ((uint64_t)mid2Lo << 32) | (uint32_t)b00;

  *productHi = pHi;
  return pLo;
#endif
}

static inline uint64_t shiftright128(const uint64_t lo, const uint64_t hi, const uint32_t dist) {
  assert(dist > 0);
  assert(dist < 64);
#if defined(RYU_HAS_64_BIT_INTRINSICS)
  return __shiftright128(lo, hi, (unsigned char) dist);
#else
  return (hi << (64 - dist)) | (lo >> dist);
#endif
}

#endif  // !RYU_HAS_UINT128

#if defined(RYU_HAS_UINT128)

static inline uint64_t mulShift64(const uint64_t m, const uint64_t* const mul, const int32_t j) {
  const ryu_uint128_t b0 = ((ryu_uint128_t) m) * mul[0];
  const ryu_uint128_t b2 = ((ryu_uint128_t) m) * mul[1];
  return (uint64_t) (((b0 >> 64) + b2) >> (j - 64));
}

static inline uint64_t mulShiftAll64(const uint64_t m, const uint64_t* const mul, const int32_t j,
  uint64_t* const vp, uint64_t* const vm, const uint32_t mmShift) {
  *vp = mulShift64(4 * m + 2, mul, j);
  *vm = mulShift64(4 * m - 1 - mmShift, mul, j);
  return mulShift64(4 * m, mul, j);
}

#else  // !RYU_HAS_UINT128

static inline uint64_t mulShift64(const uint64_t m, const uint64_t* const mul, const int32_t j) {
  // m is maximum 55 bits
  uint64_t high1;                                   // 128
  const uint64_t low1 = umul128(m, mul[1], &high1); // 64
  uint64_t high0;                                   // 64
  umul128(m, mul[0], &high0);                       // 0
  const uint64_t sum = high0 + low1;
  if (sum < high0) {
    ++high1; // overflow into high1
  }
  return shiftright128(sum, high1, (uint32_t)(j - 64));
}

#if defined(RYU_HAS_64_BIT_INTRINSICS)

static inline uint64_t mulShiftAll64(const uint64_t m, const uint64_t* const mul, const int32_t j,
  uint64_t* const vp, uint64_t* const vm, const uint32_t mmShift) {
  *vp = mulShift64(4 * m + 2, mul, j);
  *vm = mulShift64(4 * m - 1 - mmShift, mul, j);
  return mulShift64(4 * m, mul, j);
}

#else

// This is faster if we don't have a 64x64->128-bit multiplication.
static inline uint64_t mulShiftAll64(uint64_t m, const uint64_t* const mul, const int32_t j,
  uint64_t* const vp, uint64_t* const vm, const uint32_t mmShift) {
  m <<= 1;
  // m is maximum 55 bits
  uint64_t tmp;
  const uint64_t lo = umul128(m, mul[0], &tmp);
  uint64_t hi;
  const uint64_t mid = tmp + umul128(m, mul[1], &hi);
  hi += mid < tmp; // overflow into hi

  const uint64_t lo2 = lo + mul[0];
  const uint64_t mid2 = mid + mul[1] + (lo2 < lo);
  const uint64_t hi2 = hi + (mid2 < mid);
  *vp = shiftright128(mid2, hi2, (uint32_t)(j - 64 - 1));

  if (mmShift == 1) {
    const uint64_t lo3 = lo - mul[0];
    const uint64_t mid3 = mid - mul[1] - (lo3 > lo);
    const uint64_t hi3 = hi - (mid3 > mid);
    *vm = shiftright128(mid3, hi3, (uint32_t)(j - 64 - 1));
  } else {
    const uint64_t lo3 = lo + lo;
    const uint64_t mid3 = mid + mid + (lo3 < lo);
    const uint64_t hi3 = hi + hi + (mid3 < mid);
    const uint64_t lo4 = lo3 - mul[0];
    const uint64_t mid4 = mid3 - mul[1] - (lo4 > lo3);
    const uint64_t hi4 = hi3 - (mid4 > mid3);
    *vm = shiftright128(mid4, hi4, (uint32_t)(j - 64));
  }

  return shiftright128(mid, hi, (uint32_t)(j - 64 - 1));
}

#endif  // RYU_HAS_64_BIT_INTRINSICS
#endif  // RYU_HAS_UINT128

/* ---------------------------------------------------------------------------
   Small helpers.
   --------------------------------------------------------------------------- */

/* Division by small constants is written as an explicit reciprocal
   multiply rather than 'x / N'.  Compilers strength-reduce 'x / N' to a
   multiply at -O2, but profile-guided optimisation can mark the
   digit-removal loop body as cold (the PGO training workload mostly
   produces 16- to 17-digit results, so the loop runs only a couple of
   times) and then choose the much shorter hardware divide instruction
   instead, costing tens of cycles per iteration. */

static inline uint64_t umulh(const uint64_t a, const uint64_t b) {
#if defined(RYU_HAS_UINT128)
  return (uint64_t)(((ryu_uint128_t)a * b) >> 64);
#else
  uint64_t hi;
  (void)umul128(a, b, &hi);
  return hi;
#endif
}

static inline uint64_t div5(const uint64_t x)   { return umulh(x, 0xCCCCCCCCCCCCCCCDu) >> 2; }
static inline uint64_t div10(const uint64_t x)  { return umulh(x, 0xCCCCCCCCCCCCCCCDu) >> 3; }
static inline uint64_t div100(const uint64_t x) { return umulh(x >> 2, 0x28F5C28F5C28F5C3u) >> 2; }
static inline uint64_t div1e8(const uint64_t x) { return umulh(x, 0xABCC77118461CEFDu) >> 26; }

// Returns e == 0 ? 1 : ceil(log_2(5^e)); requires 0 <= e <= 3528.
static inline int32_t pow5bits(const int32_t e) {
  // This approximation works up to the point that the multiplication overflows at e = 3529.
  // If the multiplication were done in 64 bits, it would fail at 5^4004 which is just greater
  // than 2^9297.
  assert(e >= 0);
  assert(e <= 3528);
  return (int32_t) (((((uint32_t) e) * 1217359) >> 19) + 1);
}

// Returns floor(log_10(2^e)); requires 0 <= e <= 1650.
static inline uint32_t log10Pow2(const int32_t e) {
  // The first value this approximation fails for is 2^1651 which is just greater than 10^297.
  assert(e >= 0);
  assert(e <= 1650);
  return (((uint32_t) e) * 78913) >> 18;
}

// Returns floor(log_10(5^e)); requires 0 <= e <= 2620.
static inline uint32_t log10Pow5(const int32_t e) {
  // The first value this approximation fails for is 5^2621 which is just greater than 10^1832.
  assert(e >= 0);
  assert(e <= 2620);
  return (((uint32_t) e) * 732923) >> 20;
}

static inline uint32_t pow5Factor(uint64_t value) {
  const uint64_t m_inv_5 = 14757395258967641293u; // 5 * m_inv_5 = 1 (mod 2^64)
  const uint64_t n_div_5 = 3689348814741910323u;  // #{ n | n = 0 (mod 2^64) } = 2^64 / 5
  uint32_t count = 0;
  for (;;) {
    assert(value != 0);
    value *= m_inv_5;
    if (value > n_div_5)
      break;
    ++count;
  }
  return count;
}

// Returns true if value is divisible by 5^p.
static inline bool multipleOfPowerOf5(const uint64_t value, const uint32_t p) {
  // I tried a case distinction on p, but there was no performance difference.
  return pow5Factor(value) >= p;
}

// Returns true if value is divisible by 2^p.
static inline bool multipleOfPowerOf2(const uint64_t value, const uint32_t p) {
  assert(value != 0);
  assert(p < 64);
  // __builtin_ctzll doesn't appear to be faster here.
  return (value & ((1ull << p) - 1)) == 0;
}

static inline uint32_t decimalLength17(const uint64_t v) {
  // This is slightly faster than a loop.
  // The average output length is 16.38 digits, so we check high-to-low.
  // Function precondition: v is not an 18, 19, or 20-digit number.
  // (17 digits are sufficient for round-tripping.)
  assert(v < 100000000000000000L);
  if (v >= 10000000000000000L) { return 17; }
  if (v >= 1000000000000000L) { return 16; }
  if (v >= 100000000000000L) { return 15; }
  if (v >= 10000000000000L) { return 14; }
  if (v >= 1000000000000L) { return 13; }
  if (v >= 100000000000L) { return 12; }
  if (v >= 10000000000L) { return 11; }
  if (v >= 1000000000L) { return 10; }
  if (v >= 100000000L) { return 9; }
  if (v >= 10000000L) { return 8; }
  if (v >= 1000000L) { return 7; }
  if (v >= 100000L) { return 6; }
  if (v >= 10000L) { return 5; }
  if (v >= 1000L) { return 4; }
  if (v >= 100L) { return 3; }
  if (v >= 10L) { return 2; }
  return 1;
}

/* ---------------------------------------------------------------------------
   Core Ryū algorithm: convert (ieeeMantissa, ieeeExponent) to the shortest
   decimal mantissa * 10^exponent that round-trips.
   --------------------------------------------------------------------------- */

// A floating decimal representing m * 10^e.
typedef struct floating_decimal_64 {
  uint64_t mantissa;
  // Decimal exponent's range is -324 to 308
  // inclusive, and can fit in a short if needed.
  int32_t exponent;
} floating_decimal_64;

static inline floating_decimal_64 d2d(const uint64_t ieeeMantissa, const uint32_t ieeeExponent) {
  int32_t e2;
  uint64_t m2;
  if (ieeeExponent == 0) {
    // We subtract 2 so that the bounds computation has 2 additional bits.
    e2 = 1 - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS - 2;
    m2 = ieeeMantissa;
  } else {
    e2 = (int32_t) ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS - 2;
    m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
  }
  const bool even = (m2 & 1) == 0;
  const bool acceptBounds = even;

  // Step 2: Determine the interval of valid decimal representations.
  const uint64_t mv = 4 * m2;
  // Implicit bool -> int conversion. True is 1, false is 0.
  const uint32_t mmShift = ieeeMantissa != 0 || ieeeExponent <= 1;
  // We would compute mp and mm like this:
  // uint64_t mp = 4 * m2 + 2;
  // uint64_t mm = mv - 1 - mmShift;

  // Step 3: Convert to a decimal power base using 128-bit arithmetic.
  uint64_t vr, vp, vm;
  int32_t e10;
  bool vmIsTrailingZeros = false;
  bool vrIsTrailingZeros = false;
  if (e2 >= 0) {
    // I tried special-casing q == 0, but there was no effect on performance.
    // This expression is slightly faster than max(0, log10Pow2(e2) - 1).
    const uint32_t q = log10Pow2(e2) - (e2 > 3);
    e10 = (int32_t) q;
    const int32_t k = DOUBLE_POW5_INV_BITCOUNT + pow5bits((int32_t) q) - 1;
    const int32_t i = -e2 + (int32_t) q + k;
    vr = mulShiftAll64(m2, DOUBLE_POW5_INV_SPLIT[q], i, &vp, &vm, mmShift);
    if (q <= 21) {
      // This should use q <= 22, but I think 21 is also safe. Smaller values
      // may still be safe, but it's more difficult to reason about them.
      // Only one of mp, mv, and mm can be a multiple of 5, if any.
      const uint32_t mvMod5 = ((uint32_t) mv) - 5 * ((uint32_t) div5(mv));
      if (mvMod5 == 0) {
        vrIsTrailingZeros = multipleOfPowerOf5(mv, q);
      } else if (acceptBounds) {
        // Same as min(e2 + (~mm & 1), pow5Factor(mm)) >= q
        // <=> e2 + (~mm & 1) >= q && pow5Factor(mm) >= q
        // <=> true && pow5Factor(mm) >= q, since e2 >= q.
        vmIsTrailingZeros = multipleOfPowerOf5(mv - 1 - mmShift, q);
      } else {
        // Same as min(e2 + 1, pow5Factor(mp)) >= q.
        vp -= multipleOfPowerOf5(mv + 2, q);
      }
    }
  } else {
    // This expression is slightly faster than max(0, log10Pow5(-e2) - 1).
    const uint32_t q = log10Pow5(-e2) - (-e2 > 1);
    e10 = (int32_t) q + e2;
    const int32_t i = -e2 - (int32_t) q;
    const int32_t k = pow5bits(i) - DOUBLE_POW5_BITCOUNT;
    const int32_t j = (int32_t) q - k;
    vr = mulShiftAll64(m2, DOUBLE_POW5_SPLIT[i], j, &vp, &vm, mmShift);
    if (q <= 1) {
      // {vr,vp,vm} is trailing zeros if {mv,mp,mm} has at least q trailing 0 bits.
      // mv = 4 * m2, so it always has at least two trailing 0 bits.
      vrIsTrailingZeros = true;
      if (acceptBounds) {
        // mm = mv - 1 - mmShift, so it has 1 trailing 0 bit iff mmShift == 1.
        vmIsTrailingZeros = mmShift == 1;
      } else {
        // mp = mv + 2, so it always has at least one trailing 0 bit.
        --vp;
      }
    } else if (q < 63) { // TODO(ulfjack): Use a tighter bound here.
      // We want to know if the full product has at least q trailing zeros.
      // We need to compute min(p2(mv), p5(mv) - e2) >= q
      // <=> p2(mv) >= q && p5(mv) - e2 >= q
      // <=> p2(mv) >= q (because -e2 >= q)
      vrIsTrailingZeros = multipleOfPowerOf2(mv, q);
    }
  }

  // Step 4: Find the shortest decimal representation in the interval of valid representations.
  int32_t removed = 0;
  uint8_t lastRemovedDigit = 0;
  uint64_t output;
  // On average, we remove ~2 digits.
  if (vmIsTrailingZeros || vrIsTrailingZeros) {
    // General case, which happens rarely (~0.7%).
    for (;;) {
      const uint64_t vpDiv10 = div10(vp);
      const uint64_t vmDiv10 = div10(vm);
      if (vpDiv10 <= vmDiv10) {
        break;
      }
      const uint32_t vmMod10 = ((uint32_t) vm) - 10 * ((uint32_t) vmDiv10);
      const uint64_t vrDiv10 = div10(vr);
      const uint32_t vrMod10 = ((uint32_t) vr) - 10 * ((uint32_t) vrDiv10);
      vmIsTrailingZeros &= vmMod10 == 0;
      vrIsTrailingZeros &= lastRemovedDigit == 0;
      lastRemovedDigit = (uint8_t) vrMod10;
      vr = vrDiv10;
      vp = vpDiv10;
      vm = vmDiv10;
      ++removed;
    }
    if (vmIsTrailingZeros) {
      for (;;) {
        const uint64_t vmDiv10 = div10(vm);
        const uint32_t vmMod10 = ((uint32_t) vm) - 10 * ((uint32_t) vmDiv10);
        if (vmMod10 != 0) {
          break;
        }
        const uint64_t vpDiv10 = div10(vp);
        const uint64_t vrDiv10 = div10(vr);
        const uint32_t vrMod10 = ((uint32_t) vr) - 10 * ((uint32_t) vrDiv10);
        vrIsTrailingZeros &= lastRemovedDigit == 0;
        lastRemovedDigit = (uint8_t) vrMod10;
        vr = vrDiv10;
        vp = vpDiv10;
        vm = vmDiv10;
        ++removed;
      }
    }
    if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0) {
      // Round even if the exact number is .....50..0.
      lastRemovedDigit = 4;
    }
    // We need to take vr + 1 if vr is outside bounds or we need to round up.
    output = vr + ((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5);
  } else {
    // Specialized for the common case (~99.3%). Percentages below are relative to this.
    bool roundUp = false;
    const uint64_t vpDiv100 = div100(vp);
    const uint64_t vmDiv100 = div100(vm);
    if (vpDiv100 > vmDiv100) { // Optimization: remove two digits at a time (~86.2%).
      const uint64_t vrDiv100 = div100(vr);
      const uint32_t vrMod100 = ((uint32_t) vr) - 100 * ((uint32_t) vrDiv100);
      roundUp = vrMod100 >= 50;
      vr = vrDiv100;
      vp = vpDiv100;
      vm = vmDiv100;
      removed += 2;
    }
    // Loop iterations below (approximately), without optimization above:
    // 0: 0.03%, 1: 13.8%, 2: 70.6%, 3: 14.0%, 4: 1.40%, 5: 0.14%, 6+: 0.02%
    // Loop iterations below (approximately), with optimization above:
    // 0: 70.6%, 1: 27.8%, 2: 1.40%, 3: 0.14%, 4+: 0.02%
    for (;;) {
      const uint64_t vpDiv10 = div10(vp);
      const uint64_t vmDiv10 = div10(vm);
      if (vpDiv10 <= vmDiv10) {
        break;
      }
      const uint64_t vrDiv10 = div10(vr);
      const uint32_t vrMod10 = ((uint32_t) vr) - 10 * ((uint32_t) vrDiv10);
      roundUp = vrMod10 >= 5;
      vr = vrDiv10;
      vp = vpDiv10;
      vm = vmDiv10;
      ++removed;
    }
    // We need to take vr + 1 if vr is outside bounds or we need to round up.
    output = vr + (vr == vm || roundUp);
  }
  const int32_t exp = e10 + removed;

  floating_decimal_64 fd;
  fd.exponent = exp;
  fd.mantissa = output;
  return fd;
}

static inline bool d2d_small_int(const uint64_t ieeeMantissa, const uint32_t ieeeExponent,
  floating_decimal_64* const v) {
  const uint64_t m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
  const int32_t e2 = (int32_t) ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;

  if (e2 > 0) {
    // f = m2 * 2^e2 >= 2^53 is an integer.
    // Ignore this case for now.
    return false;
  }

  if (e2 < -52) {
    // f < 1.
    return false;
  }

  // Since 2^52 <= m2 < 2^53 and 0 <= -e2 <= 52: 1 <= f = m2 / 2^-e2 < 2^53.
  // Test if the lower -e2 bits of the significand are 0, i.e. whether the fraction is 0.
  const uint64_t mask = (1ull << -e2) - 1;
  const uint64_t fraction = m2 & mask;
  if (fraction != 0) {
    return false;
  }

  // f is an integer in the range [1, 2^53).
  // Note: mantissa might contain trailing (decimal) 0's.
  // Note: since 2^53 < 10^16, there is no need to adjust decimalLength17().
  v->mantissa = m2 >> -e2;
  v->exponent = 0;
  return true;
}

/* ---------------------------------------------------------------------------
   Digit emission.

   This is the inner part of upstream's to_chars(), simplified to write
   contiguous digits into 'buf' (no sign, no decimal point) with a
   trailing nul, returning the digit count.
   --------------------------------------------------------------------------- */

static inline uint32_t write_digits(uint64_t output, char* const result) {
  const uint32_t olength = decimalLength17(output);

  // Print the decimal digits.
  // The following code is equivalent to:
  // for (uint32_t i = 0; i < olength - 1; ++i) {
  //   const uint32_t c = output % 10; output /= 10;
  //   result[olength - 1 - i] = (char) ('0' + c);
  // }
  // result[0] = '0' + output % 10;

  uint32_t i = 0;
  // We prefer 32-bit operations, even on 64-bit platforms.
  // We have at most 17 digits, and uint32_t can store 9 digits.
  // If output doesn't fit into uint32_t, we cut off 8 digits,
  // so the rest will fit into uint32_t.
  if ((output >> 32) != 0) {
    // Expensive 64-bit division.
    const uint64_t q = div1e8(output);
    uint32_t output2 = ((uint32_t) output) - 100000000 * ((uint32_t) q);
    output = q;

    const uint32_t c = output2 % 10000;
    output2 /= 10000;
    const uint32_t d = output2 % 10000;
    const uint32_t c0 = (c % 100) << 1;
    const uint32_t c1 = (c / 100) << 1;
    const uint32_t d0 = (d % 100) << 1;
    const uint32_t d1 = (d / 100) << 1;
    memcpy(result + olength - 2, DIGIT_TABLE + c0, 2);
    memcpy(result + olength - 4, DIGIT_TABLE + c1, 2);
    memcpy(result + olength - 6, DIGIT_TABLE + d0, 2);
    memcpy(result + olength - 8, DIGIT_TABLE + d1, 2);
    i += 8;
  }
  uint32_t output2 = (uint32_t) output;
  while (output2 >= 10000) {
#ifdef __clang__ // https://bugs.llvm.org/show_bug.cgi?id=38217
    const uint32_t c = output2 - 10000 * (output2 / 10000);
#else
    const uint32_t c = output2 % 10000;
#endif
    output2 /= 10000;
    const uint32_t c0 = (c % 100) << 1;
    const uint32_t c1 = (c / 100) << 1;
    memcpy(result + olength - i - 2, DIGIT_TABLE + c0, 2);
    memcpy(result + olength - i - 4, DIGIT_TABLE + c1, 2);
    i += 4;
  }
  if (output2 >= 100) {
    const uint32_t c = (output2 % 100) << 1;
    output2 /= 100;
    memcpy(result + olength - i - 2, DIGIT_TABLE + c, 2);
    i += 2;
  }
  if (output2 >= 10) {
    const uint32_t c = output2 << 1;
    memcpy(result + olength - i - 2, DIGIT_TABLE + c, 2);
  } else {
    result[0] = (char) ('0' + output2);
  }
  result[olength] = '\0';
  return olength;
}

/* ---------------------------------------------------------------------------
   Public entry point.
   --------------------------------------------------------------------------- */

int
_Py_ryu_dtoa(double d, char *buf, int *decpt, int *sign)
{
    uint64_t bits;
    memcpy(&bits, &d, sizeof(double));

    const bool ieeeSign =
        ((bits >> (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS)) & 1) != 0;
    const uint64_t ieeeMantissa =
        bits & ((1ull << DOUBLE_MANTISSA_BITS) - 1);
    const uint32_t ieeeExponent =
        (uint32_t)((bits >> DOUBLE_MANTISSA_BITS)
                   & ((1u << DOUBLE_EXPONENT_BITS) - 1));

    *sign = ieeeSign ? 1 : 0;

    if (ieeeExponent == ((1u << DOUBLE_EXPONENT_BITS) - 1u)) {
        *decpt = 9999;
        if (ieeeMantissa != 0) {
            memcpy(buf, "NaN", 4);
            return 3;
        }
        memcpy(buf, "Infinity", 9);
        return 8;
    }
    if (ieeeExponent == 0 && ieeeMantissa == 0) {
        buf[0] = '0';
        buf[1] = '\0';
        *decpt = 1;
        return 1;
    }

    floating_decimal_64 v;
    if (d2d_small_int(ieeeMantissa, ieeeExponent, &v)) {
        /* Strip trailing decimal zeros from small integers. */
        for (;;) {
            const uint64_t q = div10(v.mantissa);
            const uint32_t r = ((uint32_t)v.mantissa) - 10 * ((uint32_t)q);
            if (r != 0) {
                break;
            }
            v.mantissa = q;
            ++v.exponent;
        }
    }
    else {
        v = d2d(ieeeMantissa, ieeeExponent);
    }

    const uint32_t olength = write_digits(v.mantissa, buf);
    *decpt = v.exponent + (int32_t)olength;
    return (int)olength;
}

#endif  /* _PY_SHORT_FLOAT_REPR == 1 */
