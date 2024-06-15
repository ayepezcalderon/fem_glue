import unittest
from fem_glue.geometry import Point

class TestPoint(unittest.TestCase):
    def setUp(self):
        self.point = Point(3, 4, 5)

    def test_norm(self):
        result = self.point.norm()
        expected = (3**2 + 4**2 + 5**2)**0.5
        self.assertEqual(result, expected)

    def test_add(self):
        other = Point(1, 2, 3)
        result = self.point + other
        expected = Point(4, 6, 8)
        self.assertEqual(result, expected)

    def test_subtract(self):
        other = Point(1, 2, 3)
        result = self.point - other
        expected = Point(2, 2, 2)
        self.assertEqual(result, expected)

    def test_multiply(self):
        factor = 2
        result = self.point * factor
        expected = Point(6, 8, 10)
        self.assertEqual(result, expected)

    def test_divide(self):
        divisor = 2
        result = self.point / divisor
        expected = Point(1.5, 2, 2.5)
        self.assertEqual(result, expected)

    def test_pow(self):
        exp = 2
        result = self.point ** exp



if __name__ == '__main__':
    unittest.main()

