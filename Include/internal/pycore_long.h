#ifndef Py_INTERNAL_LONG_H
#define Py_INTERNAL_LONG_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_global_objects.h"  // _PY_NSMALLNEGINTS
#include "pycore_runtime.h"       // _PyRuntime

/*
 * Default int base conversion size limitation.
 *
 * Chosen such that this isn't wildly slow on modern hardware:
 * % python -m timeit -s 's = "1"*2000; v = int(s)' 'str(int(s))'
 * 2000 loops, best of 5: 100 usec per loop
 *
 * 2000 decimal digits fits a ~6643 bit number.
 */
#define _PY_LONG_DEFAULT_MAX_BASE10_DIGITS 2000
/*
 * Threshold for max digits check.  For performance reasons int() and
 * int.__str__ don't checks values that are smaller than this
 * threshold.  Acts as a guaranteed minimum size limit for bignums that
 * applications can expect from CPython.
 *
 * % python -m timeit -s 's = "1"*333; v = int(s)' 'str(int(s))'
 * 100000 loops, best of 5: 3.94 usec per loop
 *
 * 333 decimal digits fits a ~1106 bit number.
 */
#define _PY_LONG_MAX_BASE10_DIGITS_THRESHOLD 333

#if ((_PY_LONG_DEFAULT_MAX_BASE10_DIGITS != 0) && \
   (_PY_LONG_DEFAULT_MAX_BASE10_DIGITS < _PY_LONG_MAX_BASE10_DIGITS_THRESHOLD))
# error "_PY_LONG_DEFAULT_MAX_BASE10_DIGITS smaller than threshold."
#endif


/* runtime lifecycle */

extern PyStatus _PyLong_InitTypes(PyInterpreterState *);
extern void _PyLong_FiniTypes(PyInterpreterState *interp);


/* other API */

#define _PyLong_SMALL_INTS _Py_SINGLETON(small_ints)

// _PyLong_GetZero() and _PyLong_GetOne() must always be available
// _PyLong_FromUnsignedChar must always be available
#if _PY_NSMALLPOSINTS < 257
#  error "_PY_NSMALLPOSINTS must be greater than or equal to 257"
#endif

// Return a borrowed reference to the zero singleton.
// The function cannot return NULL.
static inline PyObject* _PyLong_GetZero(void)
{ return (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS]; }

// Return a borrowed reference to the one singleton.
// The function cannot return NULL.
static inline PyObject* _PyLong_GetOne(void)
{ return (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS+1]; }

static inline PyObject* _PyLong_FromUnsignedChar(unsigned char i)
{
    return Py_NewRef((PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS+i]);
}

PyObject *_PyLong_Add(PyLongObject *left, PyLongObject *right);
PyObject *_PyLong_Multiply(PyLongObject *left, PyLongObject *right);
PyObject *_PyLong_Subtract(PyLongObject *left, PyLongObject *right);

int _PyLong_AssignValue(PyObject **target, Py_ssize_t value);

/* Used by Python/mystrtoul.c, _PyBytes_FromHex(),
   _PyBytes_DecodeEscape(), etc. */
PyAPI_DATA(unsigned char) _PyLong_DigitValue[256];

/* Format the object based on the format_spec, as defined in PEP 3101
   (Advanced String Formatting). */
PyAPI_FUNC(int) _PyLong_FormatAdvancedWriter(
    _PyUnicodeWriter *writer,
    PyObject *obj,
    PyObject *format_spec,
    Py_ssize_t start,
    Py_ssize_t end);

PyAPI_FUNC(int) _PyLong_FormatWriter(
    _PyUnicodeWriter *writer,
    PyObject *obj,
    int base,
    int alternate);

PyAPI_FUNC(char*) _PyLong_FormatBytesWriter(
    _PyBytesWriter *writer,
    char *str,
    PyObject *obj,
    int base,
    int alternate);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_LONG_H */
