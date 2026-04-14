import math
import sys
import unittest
from test import support
from test.support.hypothesis_helper import hypothesis

floats = hypothesis.strategies.floats
given = hypothesis.given
example = hypothesis.example


@unittest.skipUnless(getattr(sys, 'float_repr_style', '') == 'short',
                     "applies only when using short float repr style")
class ShortFloatReprProperties(unittest.TestCase):

    @given(x=floats())
    @example(x=0.0)
    @example(x=-0.0)
    @example(x=5e-324)
    @example(x=sys.float_info.min)
    @example(x=sys.float_info.max)
    def test_repr_round_trip(self, x):
        s = repr(x)
        if math.isnan(x):
            self.assertTrue(math.isnan(float(s)))
        else:
            self.assertEqual(float(s), x)
            self.assertEqual(str(x), s)

    @given(x=floats(allow_nan=False, allow_infinity=False))
    @example(x=2.0**-24)
    @example(x=1e-323)
    @example(x=2.0**89)
    @example(x=2.0**-44)
    @example(x=1.0)
    @example(x=0.0)
    def test_repr_is_shortest(self, x):
        # Steele & White criterion (1): no shorter decimal string round-trips.
        s = repr(x)
        mant = s.partition('e')[0]
        digits = ''.join(c for c in mant if c.isdigit())
        if '.' in mant:
            digits = digits.rstrip('0')
        digits = digits.lstrip('0') or '0'
        n = len(digits)
        if n > 1:
            shorter = format(x, f'.{n - 1}g')
            if float(shorter) == x:
                self.fail(f'repr({x.hex()}) = {s!r} ({n} digits) '
                          f'but {shorter!r} ({n - 1} digits) also '
                          f'round-trips')

    @support.cpython_only
    @given(x=floats())
    @example(x=2.0**-24)
    @example(x=2.0**-25)
    @example(x=5e-324)
    @example(x=2.98023223876953125e-8)
    @example(x=float.fromhex('0x1.0F0CF064DD592p+132'))
    @example(x=4.940656e-318)
    @example(x=9.0608011534336e15)
    @example(x=4.708356024711512e18)
    @example(x=-0.0)
    def test_ryu_matches_dtoa(self, x):
        # Ryu and the reference dtoa must produce identical
        # (digits, decpt, sign) triples for the shortest representation.
        from test.support import import_helper
        _testinternalcapi = import_helper.import_module('_testinternalcapi')
        mismatch = _testinternalcapi.compare_ryu_dtoa(x)
        if mismatch is not None and not math.isnan(x):
            self.fail(f'ryu/dtoa mismatch for {x.hex()} ({x!r}): '
                      f'dtoa={mismatch[0]} ryu={mismatch[1]}')

    @given(re=floats(), im=floats())
    @example(re=-0.0, im=-0.0)
    @example(re=0.0, im=-0.0)
    @example(re=math.inf, im=-math.inf)
    @example(re=1e308, im=1e-308)
    @example(re=5e-324, im=5e-324)
    def test_complex_repr_round_trip(self, re, im):
        c = complex(re, im)
        s = repr(c)
        c2 = complex(s)
        if math.isnan(re):
            self.assertTrue(math.isnan(c2.real))
        else:
            self.assertEqual(c2.real, re)
        if math.isnan(im):
            self.assertTrue(math.isnan(c2.imag))
        else:
            self.assertEqual(c2.imag, im)


if __name__ == '__main__':
    unittest.main()
