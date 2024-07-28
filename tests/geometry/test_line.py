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
