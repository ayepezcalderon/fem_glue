import unittest
from fem_glue.geometry import Point, Line
from fem_glue._config import CONFIG


class TestLine(unittest.TestCase):
    def setUp(self):
        self.line1 = Line([Point([0, 0, 0]), Point([1, 1, 0])])
        self.line2 = Line([Point([0, 1, 0]), Point([1, 0, 0])])
        self.line3 = Line([Point([2, 2, 0]), Point([3, 3, 0])])
        self.line4 = Line([Point([0, 0, 0]), Point([0, 1, 0])])
        self.line5 = Line([Point([0, 1, 0]), Point([1, 2, 0])])
        self.line6 = Line([Point([0.5, 0.5, 0]), Point([1.5, 1.5, 0])])
        self.line7 = Line([Point([10, 1, 0]), Point([10, 0, 0])])

    def test_length(self):
        self.assertEqual(self.line1.length(), round(2**0.5, CONFIG.precision))

    def test_intersect(self):
        # Test intersection of two lines that intersect at a point
        # Non-colinear
        self.assertEqual(self.line1.intersect(self.line2), Point([0.5, 0.5, 0]))
        # Colinear
        self.assertEqual(self.line1.intersect(self.line4), Point([0, 0, 0]))

        # Test intersection of two lines that do not intersect
        # Parallel but not colinear
        self.assertIsNone(self.line1.intersect(self.line5))
        # Colinear but non-overlapping
        self.assertIsNone(self.line1.intersect(self.line3))
        # Non-parallel but non-overlapping
        self.assertIsNone(self.line1.intersect(self.line7))

        # Test intersection of two lines that intersect at a line segment
        self.assertEqual(
            self.line1.intersect(self.line6),
            Line([Point([0.5, 0.5, 0]), Point([1, 1, 0])]),
        )

    def test_normalize(self):
        normalized_line = self.line1.normalize()
        self.assertEqual(normalized_line.length(), 1)


class TestLineOperations(unittest.TestCase):
    def setUp(self):
        self.line = Line([Point([3, 4, 5]), Point([6, 8, 10])])

    def test_add(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.line + scalar, Line([Point([5.5, 6.5, 7.5]), Point([8.5, 10.5, 12.5])])
        )

        # With sequence
        seq = [1, 2.5, 3]
        self.assertEqual(
            self.line + seq, Line([Point([4, 6.5, 8]), Point([7, 10.5, 13])])
        )

    def test_subtract(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.line - scalar, Line([Point([0.5, 1.5, 2.5]), Point([3.5, 5.5, 7.5])])
        )

        # With sequence
        seq = [1, 2.5, 3]
        self.assertEqual(
            self.line - seq, Line([Point([2, 1.5, 2]), Point([5, 5.5, 7])])
        )

    def test_multiply(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.line * scalar, Line([Point([7.5, 10, 12.5]), Point([15, 20, 25])])
        )

        # With sequence
        seq = [1, 2.5, 3]
        self.assertEqual(
            self.line * seq, Line([Point([3, 10, 15]), Point([6, 20, 30])])
        )

    def test_divide(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.line / scalar, Line([Point([1.2, 1.6, 2]), Point([2.4, 3.2, 4])])
        )

        # With sequence
        seq = [1, 2, 4]
        self.assertEqual(
            self.line / seq, Line([Point([3, 2, 1.25]), Point([6, 4, 2.5])])
        )

    def test_pow(self):
        exp = 2
        self.assertEqual(
            self.line**exp, Line([Point([9, 16, 25]), Point([36, 64, 100])])
        )
