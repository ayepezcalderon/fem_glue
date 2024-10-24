import re
import unittest

from fem_glue._config import CONFIG
from fem_glue.geometry import Line, Point, Polyline


class TestDefaultClosedNonIntersecting(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point([0, 0, 0]),
            Point([1, 0, 0]),
            Point([1, 2, 0]),
            Point([0, 2, 0]),
            Point([0, 0, 0]),
        ]
        self.lines = (
            Line([self.points[0], self.points[1]]),
            Line([self.points[1], self.points[2]]),
            Line([self.points[2], self.points[3]]),
            Line([self.points[3], self.points[0]]),
        )
        self.polyline = Polyline(self.points)

    def test_init_with_points(self):
        polyline = Polyline(self.points)
        self.assertEqual(polyline._elements, self.lines)
        self.assertEqual(polyline.points, self.points[:-1])

    def test_init_with_lines(self):
        polyline = Polyline(self.lines)
        self.assertEqual(polyline._elements, self.lines)
        self.assertEqual(polyline.points, self.points[:-1])

    def test_init_with_mixed_elements(self):
        with self.assertRaises(TypeError):
            Polyline([self.points[0], self.lines[0]])  # type: ignore

    def test_len(self):
        self.assertEqual(len(self.polyline), 4)

    def test_perimeter(self):
        self.assertEqual(self.polyline.perimeter(), 6)

    def test_is_closed(self):
        self.assertTrue(self.polyline.is_closed())


class TestDefaultOpenSelfIntersecting(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point([0, 0, 0]),
            Point([8, 0, 0]),
            Point([4, 3, 0]),
            Point([4, -3, 0]),
            Point([4, 0, 0]),
        ]
        self.lines = (
            Line([self.points[0], self.points[1]]),
            Line([self.points[1], self.points[2]]),
            Line([self.points[2], self.points[3]]),
            Line([self.points[3], self.points[4]]),
        )
        self.polyline = Polyline(self.points)

    def test_init_with_points(self):
        polyline = Polyline(self.points)
        self.assertEqual(polyline._elements, self.lines)
        self.assertEqual(polyline.points, self.points)

    def test_init_with_lines(self):
        polyline = Polyline(self.lines)
        self.assertEqual(polyline._elements, self.lines)
        self.assertEqual(polyline.points, self.points)

    def test_len(self):
        self.assertEqual(len(self.polyline), 4)

    def test_perimeter(self):
        self.assertAlmostEqual(
            self.polyline.perimeter(), 8 + 5 + 6 + 3, places=CONFIG.precision
        )

    def test_is_closed(self):
        self.assertFalse(self.polyline.is_closed())

    def test_get_self_intersections(self):
        points, lines = self.polyline.get_self_intersections()

        self.assertEqual(points, [self.points[4]])

        self.assertEqual(len(lines), 1)
        self.assertTrue(self.points[-1] in lines[0])
        self.assertTrue(self.points[-2] in lines[0])


class TestSpecifiedClosedNonIntersecting(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point([0, 0, 0]),
            Point([1, 0, 0]),
            Point([1, 2, 0]),
            Point([0, 2, 0]),
            Point([0, 0, 0]),
        ]
        self.lines = (
            Line([self.points[0], self.points[1]]),
            Line([self.points[1], self.points[2]]),
            Line([self.points[2], self.points[3]]),
            Line([self.points[3], self.points[0]]),
        )
        self.polyline = Polyline(self.points)

    def test_init_with_points(self):
        polyline = Polyline(self.points)
        self.assertEqual(polyline._elements, self.lines)
        self.assertEqual(polyline.points, self.points[:-1])

    def test_init_with_lines(self):
        polyline = Polyline(self.lines)
        self.assertEqual(polyline._elements, self.lines)
        self.assertEqual(polyline.points, self.points[:-1])

    def test_init_with_mixed_elements(self):
        with self.assertRaises(TypeError):
            Polyline([self.points[0], self.lines[0]])  # type: ignore

    def test_len(self):
        self.assertEqual(len(self.polyline), 4)


class TestInitErrors(unittest.TestCase):
    def test_bad_type_init_elements(self):
        points = [
            [0, 0, 0],
            [8, 0, 0],
            [4, 3, 0],
            [4, -3, 0],
            [4, 0, 0],
        ]

        with self.assertRaisesRegex(
            TypeError,
            re.escape("The elements must be either Points or Lines."),
        ):
            Polyline(points)  # type: ignore

    def test_not_enough_points_init(self):
        points = [
            Point([0, 0, 0]),
            Point([8, 0, 0]),
        ]

        with self.assertRaisesRegex(
            ValueError,
            re.escape(f"A {Polyline.__name__} must have at least 3 points."),
        ):
            Polyline(points)

    def test_not_enough_lines_init(self):
        points = [
            Point([0, 0, 0]),
            Point([8, 0, 0]),
        ]
        lines = [Line(points)]

        with self.assertRaisesRegex(
            ValueError,
            re.escape(f"A {Polyline.__name__} must have at least 2 lines."),
        ):
            Polyline(lines)

    def test_specified_non_intersecting_error(self):
        points = [
            Point([0, 0, 0]),
            Point([8, 0, 0]),
            Point([4, 3, 0]),
            Point([4, -3, 0]),
            Point([4, 0, 0]),
        ]

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "The polyline is self intersecting. If this should not raise an error, "
                "set 'strict_non_intersecting' to False."
            ),
        ):
            Polyline(points, strict_non_intersecting=True)

    def test_specified_lines_not_connected_error(self):
        points = [
            Point([0, 0, 0]),
            Point([8, 0, 0]),
            Point([4, 3, 0]),
            Point([4, -3, 0]),
            Point([4, 0, 0]),
        ]
        lines = (
            Line([points[0], points[1]]),
            Line([points[1], points[2]]),
            Line([points[3], points[4]]),
        )

        with self.assertRaisesRegex(
            ValueError,
            re.escape("Line '1' is not connected with line '2'."),
        ):
            Polyline(lines)


class TestArithmeticOperations(unittest.TestCase):
    def setUp(self):
        self.polyline = Polyline(
            [Point([3, 4, 5]), Point([6, 8, 10]), Point([9, 12, 15])]
        )

    def test_add(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.polyline + scalar,
            Polyline(
                [
                    Point([5.5, 6.5, 7.5]),
                    Point([8.5, 10.5, 12.5]),
                    Point([11.5, 14.5, 17.5]),
                ]
            ),
        )

        # With sequence
        seq = [1, 2.5, 3]
        self.assertEqual(
            self.polyline + seq,
            Polyline([Point([4, 6.5, 8]), Point([7, 10.5, 13]), Point([10, 14.5, 18])]),
        )

    def test_subtract(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.polyline - scalar,
            Polyline(
                [
                    Point([0.5, 1.5, 2.5]),
                    Point([3.5, 5.5, 7.5]),
                    Point([6.5, 9.5, 12.5]),
                ]
            ),
        )

        # With sequence
        seq = [1, 2.5, 3]
        self.assertEqual(
            self.polyline - seq,
            Polyline([Point([2, 1.5, 2]), Point([5, 5.5, 7]), Point([8, 9.5, 12])]),
        )

    def test_multiply(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.polyline * scalar,
            Polyline(
                [Point([7.5, 10, 12.5]), Point([15, 20, 25]), Point([22.5, 30, 37.5])]
            ),
        )

        # With sequence
        seq = [1, 2.5, 3]
        self.assertEqual(
            self.polyline * seq,
            Polyline([Point([3, 10, 15]), Point([6, 20, 30]), Point([9, 30, 45])]),
        )

    def test_divide(self):
        # With scalar
        scalar = 2.5
        self.assertEqual(
            self.polyline / scalar,
            Polyline(
                [Point([1.2, 1.6, 2]), Point([2.4, 3.2, 4]), Point([3.6, 4.8, 6])]
            ),
        )

        # With sequence
        seq = [1, 2, 4]
        self.assertEqual(
            self.polyline / seq,
            Polyline([Point([3, 2, 1.25]), Point([6, 4, 2.5]), Point([9, 6, 3.75])]),
        )

    def test_pow(self):
        exp = 2
        self.assertEqual(
            self.polyline**exp,
            Polyline([Point([9, 16, 25]), Point([36, 64, 100]), Point([81, 144, 225])]),
        )
