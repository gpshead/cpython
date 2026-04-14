#ifndef Py_INTERNAL_RYU_H
#define Py_INTERNAL_RYU_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_pymath.h"        // _PY_SHORT_FLOAT_REPR

#if _PY_SHORT_FLOAT_REPR == 1

/* Space for the longest possible result of _Py_ryu_dtoa: 17 digits + nul,
   or "Infinity" + nul. */
#define _Py_RYU_DTOA_BUFSIZE 18

/* Compute the shortest decimal representation of 'd' that round-trips
   when parsed back as a double.

   Writes 1 to 17 ASCII digits into 'buf' (no sign, no decimal point,
   no exponent), null-terminated.  For NaN and infinities writes "NaN"
   or "Infinity" instead.  'buf' must have room for at least
   _Py_RYU_DTOA_BUFSIZE bytes.

   On return, *decpt is set to the position of the decimal point relative
   to the start of the digit string (9999 for NaN/Inf), and *sign is set
   to 0 for positive values and 1 for negative values.

   This produces the same (digits, decpt, sign) triple as
   _Py_dg_dtoa(d, 0, 0, decpt, sign, ...), but uses only fixed-size
   integer arithmetic and writes into a caller-provided buffer.

   Returns the number of bytes written, not counting the trailing nul. */
// Export for '_testinternalcapi' shared extension.
PyAPI_FUNC(int) _Py_ryu_dtoa(double d, char *buf, int *decpt, int *sign);


/* Space for the longest possible Python repr() of a float plus a
   trailing nul:

       sign 1 + digit 1 + '.' 1 + digits 16 + 'e' 1 + sign 1 + exp 3
       = 24 chars + nul = 25.

   Rounded up to the next 4-byte boundary. */
#define _Py_DOUBLE_REPR_BUFSIZE 28

/* Write the Python repr of 'val' (as PyOS_double_to_string with format
   code 'r' and the given Py_DTSF_* flags) into the caller-provided
   buffer 'buf' of size 'bufsize', which must be at least
   _Py_DOUBLE_REPR_BUFSIZE.  The result is null-terminated.

   Returns the number of bytes written, not counting the trailing nul. */
// Export for '_json' shared extension.
PyAPI_FUNC(Py_ssize_t) _Py_double_repr_buffered(double val, char *buf,
                                                Py_ssize_t bufsize, int flags);

#endif  // _PY_SHORT_FLOAT_REPR == 1

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_RYU_H */
