import unittest
from fem_glue.geometry import Point
from fem_glue._config import CONFIG


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.point = Point([3, 4, 5])

    def test_norm(self):
        result = self.point.norm()
        expected = round((3**2 + 4**2 + 5**2) ** 0.5, CONFIG.precision)
        self.assertEqual(result, expected)

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
