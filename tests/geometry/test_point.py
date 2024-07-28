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

    def test_add(self):
        # With another point
        other_point = Point([1, 2, 3])
        self.assertEqual(self.point + other_point, Point([4, 6, 8]))

        # With scalar
        scalar = 2.5
        self.assertEqual(self.point + scalar, Point([5.5, 6.5, 7.5]))

        # With list
        other_list = [1, 2, 3]
        with self.assertRaises(TypeError):
            _ = self.point + other_list

    def test_subtract(self):
        # With another point
        other_point = Point([1, 2, 3])
        self.assertEqual(self.point - other_point, Point([2, 2, 2]))

        # With scalar
        scalar = 2.5
        self.assertEqual(self.point - scalar, Point([0.5, 1.5, 2.5]))

        # With list
        other_list = [1, 2, 3]
        with self.assertRaises(TypeError):
            _ = self.point + other_list


#     def test_subtract(self):
#         # With another point
#         other_point = Point(1, 2, 3)
#         result = self.point - other_point
#         expected = Point(2, 2, 2)
#         self.assertEqual(result, expected)
#
#         # With scalar
#         other = 2.5
#         result = self.point - other
#         expected = Point(0.5, 1.5, 2.5)
#         self.assertEqual(result, expected)
#
#     def test_multiply(self):
#         # With another Point
#         other_point = Point(1, 2, 3)
#         result = self.point * other_point
#         expected = Point(3, 8, 15)
#         self.assertEqual(result, expected)
#
#         # With scalar
#         factor = 2
#         result = self.point * factor
#         expected = Point(6, 8, 10)
#         self.assertEqual(result, expected)
#
#     def test_divide(self):
#         divisor = 2
#         result = self.point / divisor
#         expected = Point(1.5, 2, 2.5)
#         self.assertEqual(result, expected)
#
#     def test_pow(self):
#         exp = 2
#         result = self.point ** exp
#         expected = Point(9, 16, 25)
#         self.assertEqual(result, expected)
#
#
#
# if __name__ == '__main__':
#     unittest.main()
#
