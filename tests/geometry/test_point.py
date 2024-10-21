import math
import unittest

import numpy as np

from fem_glue._config import CONFIG
from fem_glue.geometry import Point


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.point = Point([3, 4, 5])

    def test_normalize(self):
        normalized = self.point.normalize()

        # Check norm
        self.assertAlmostEqual(math.hypot(*normalized), 1, places=CONFIG.precision)

        # Check inverse operation
        self.assertEqual(
            (Point(list(normalized)) * self.point.norm()).round(5),
            self.point,
        )

    def test_round(self):
        unrounded = Point([3.14159, 4.14159, 5.14159])
        rounded = unrounded.round(3)

        self.assertEqual(rounded, Point([3.142, 4.142, 5.142]))
        self.assertNotEqual(rounded, Point([3.1426, 4.1426, 5.1426]))
        self.assertNotEqual(rounded, unrounded)

    def test_as_array(self):
        array_from_point = self.point.as_array()

        self.assertIsInstance(array_from_point, np.ndarray)
        self.assertEqual(array_from_point.shape, (3,))
        self.assertTrue(np.equal(array_from_point, [3, 4, 5]).all())

    def test_norm(self):
        result = self.point.norm()
        expected = round((3**2 + 4**2 + 5**2) ** 0.5, CONFIG.precision)
        self.assertEqual(result, expected)

    def test_in_keyword(self):
        for n in self.point:
            # Differs from n by less than the config tolerance
            almost_n = n + 10 ** -(CONFIG.precision + 1)

            self.assertNotIn(almost_n, self.point._elements)
            self.assertIn(almost_n, self.point)

    def test_reversed(self):
        self.assertEqual(self.point.reversed(), Point([5, 4, 3]))

    def test_math_errors(self):
        # With list with different number of entries
        other_list = [1, 2, 3, 4]
        with self.assertRaises(ValueError):
            _ = self.point + other_list

        # With list with entries with wrong type
        other_list = [1, "fads", 3]
        with self.assertRaises(TypeError):
            _ = self.point + other_list

    def test_add(self):
        # With another point
        other_point = Point([1, 2, 3])
        self.assertEqual(self.point + other_point, Point([4, 6, 8]))

        # With scalar
        scalar = 2.5
        self.assertEqual(self.point + scalar, Point([5.5, 6.5, 7.5]))

    def test_subtract(self):
        # With another point
        other_point = Point([1, 2, 3])
        self.assertEqual(self.point - other_point, Point([2, 2, 2]))

        # With scalar
        scalar = 2.5
        self.assertEqual(self.point - scalar, Point([0.5, 1.5, 2.5]))

    def test_multiply(self):
        # With another Point
        other_point = Point([1, 2, 3])
        self.assertEqual(self.point * other_point, Point([3, 8, 15]))

        # With scalar
        scalar = 2
        self.assertEqual(self.point * scalar, Point([6, 8, 10]))

    def test_divide(self):
        # With another Point
        other_point = Point([1, 2, 4])
        self.assertEqual(self.point / other_point, Point([3, 2, 1.25]))

        # With scalar
        scalar = 2
        self.assertEqual(self.point / scalar, Point([1.5, 2, 2.5]))

    def test_pow(self):
        exp = 2
        self.assertEqual(self.point**exp, Point([9, 16, 25]))
