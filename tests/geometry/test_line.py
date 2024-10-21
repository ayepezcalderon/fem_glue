import math
import re
import unittest

import numpy as np

from fem_glue._config import CONFIG
from fem_glue.geometry import Line, Point
from fem_glue.geometry._exceptions import PointNotOnShapeError, PointOnShapeError


class TestLine(unittest.TestCase):
    def setUp(self):
        self.line1 = Line([Point([0, 0, 0]), Point([1, 1, 0])])
        self.line2 = Line([Point([0, 1, 0]), Point([1, 0, 0])])
        self.line3 = Line([Point([2, 2, 0]), Point([3, 3, 0])])
        self.line4 = Line([Point([0, 0, 0]), Point([0, 1, 0])])
        self.line5 = Line([Point([0, 1, 0]), Point([1, 2, 0])])
        self.line6 = Line([Point([0.5, 0.5, 0]), Point([1.5, 1.5, 0])])
        self.line7 = Line([Point([10, 1, 0]), Point([10, 0, 0])])

    def test_init_equal_points_error(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("A line cannot be constructed from two identical points."),
        ):
            Line([Point([1, 0, 0]), Point([1, 0, 0])])

    def test_in_keyword(self):
        for p in self.line6:
            # Differs from n by less than the config tolerance
            almost_p = p + 10 ** -(CONFIG.precision + 1)

            self.assertIn(almost_p, self.line6)

    def test_length(self):
        self.assertEqual(self.line1.length(), round(2**0.5, CONFIG.precision))

    def test_as_vector(self):
        expected = np.array([1, 1, 0])
        np.testing.assert_array_equal(self.line3.as_vector(), expected)

    def test_dir_unit_vector(self):
        expected = np.array([2 ** (-1 / 2), 2 ** (-1 / 2), 0])
        np.testing.assert_array_almost_equal(self.line3.dir_unit_vector(), expected)

    def test_is_collinear(self):
        # Test collinear lines
        self.assertTrue(self.line1.is_collinear(self.line1))
        self.assertTrue(self.line1.is_collinear(self.line6))

        # Test non-collinear lines
        self.assertFalse(self.line1.is_collinear(self.line2))
        self.assertFalse(self.line1.is_collinear(self.line5))
        # Ensure that order of point-coincidence does not break functionality
        self.assertFalse(self.line1.is_collinear(self.line4))
        self.assertFalse(self.line1.is_collinear(self.line4.reversed()))
        self.assertFalse(self.line1.reversed().is_collinear(self.line4))
        self.assertFalse(self.line1.reversed().is_collinear(self.line4.reversed()))

        # Test error
        with self.assertRaises(TypeError):
            self.line1.is_collinear(self.line1._elements)  # type: ignore

    def test_is_parallel(self):
        # Test parallel lines
        self.assertTrue(self.line1.is_parallel(self.line1))
        self.assertTrue(self.line1.is_parallel(self.line6))
        self.assertTrue(self.line1.is_parallel(self.line5))

        # Test non-parallel lines
        self.assertFalse(self.line1.is_parallel(self.line2))
        self.assertFalse(self.line1.is_parallel(self.line4))

        # Test error
        with self.assertRaises(TypeError):
            self.line1.is_parallel(self.line1._elements)  # type: ignore

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

    def test_get_point_projection_on_ray(self):
        # Projection outside line
        point = Point([2, 0, 0])
        projection = self.line1.get_point_projection_on_ray(point)
        self.assertEqual(projection, Point([1, 1, 0]))

        # Projection inside line
        point = Point([1, 0, 0])
        projection = self.line1.get_point_projection_on_ray(point)
        self.assertEqual(projection, Point([0.5, 0.5, 0]))

        # Projection is on ray outside line, "self" behavior
        point = Point([2, 2, 0])
        projection = self.line1.get_point_projection_on_ray(
            point, point_is_on_ray="self"
        )
        self.assertIs(projection, point)

        # Point is on ray inside line
        # "self" behavior
        point = Point([0.5, 0.5, 0])
        projection = self.line1.get_point_projection_on_ray(
            point, point_is_on_ray="self"
        )
        self.assertIs(projection, point)
        # raise" error
        with self.assertRaises(PointOnShapeError):
            self.line1.get_point_projection_on_ray(point, point_is_on_ray="raise")

    def test_get_point_position_on_ray(self):
        ref_line = self.line1 * 2

        # Position of point on normalized coordinate system
        point = Point([3, 3, 0])
        position = ref_line.get_point_position_on_ray(point, normalized=True)
        assert isinstance(position, float)
        self.assertAlmostEqual(position, 1.5, places=CONFIG.precision)

        # Position of point on non-normalized coordinate system
        position = ref_line.get_point_position_on_ray(point, normalized=False)
        assert isinstance(position, float)
        self.assertAlmostEqual(position, 3 * math.sqrt(2), places=CONFIG.precision)

        # Point not on ray, "null" behavior
        point = Point([3, 4, 0])
        position = ref_line.get_point_position_on_ray(point, point_is_not_on_ray="null")
        self.assertIsNone(position)

        # Point not on ray, "raise" behavior
        with self.assertRaises(PointNotOnShapeError):
            _ = ref_line.get_point_position_on_ray(point, point_is_not_on_ray="raise")

    def test_get_point_projection_on_line(self):
        # Projection inside line
        point = Point([1, 0, 0])
        projection = self.line1.get_point_projection_on_line(point)
        self.assertEqual(projection, Point([0.5, 0.5, 0]))

        # Point is on line
        point = Point([0.5, 0.5, 0])
        # "self" behavior
        projection = self.line1.get_point_projection_on_line(
            point, point_is_on_line="self"
        )
        self.assertIs(projection, point)
        # "raise" behavior
        with self.assertRaises(PointOnShapeError):
            _ = self.line1.get_point_projection_on_line(point, point_is_on_line="raise")

        # Projection outside line, one point outside ray and another on ray
        points = [
            Point([3, 0, 0]),
            Point([2, 2, 0]),
        ]
        for point in points:
            # "null" behavior
            projection = self.line1.get_point_projection_on_line(
                point, projection_is_not_on_line="null"
            )
            self.assertIsNone(projection)
            # "raise" behavior
            with self.assertRaises(PointNotOnShapeError):
                _ = self.line1.get_point_projection_on_line(
                    point, projection_is_not_on_line="raise"
                )

    def test_get_shortest_line_to_point(self):
        # Standard case -> point is not on line but projection is on line
        projection_on_line_point = Point([1, 0, 0])
        shortest_line = self.line1.get_shortest_line_to_point(projection_on_line_point)
        assert isinstance(shortest_line, Line)
        self.assertEqual(shortest_line[0], Point([0.5, 0.5, 0]))
        self.assertEqual(shortest_line[1], projection_on_line_point)

        # Point not on ray and projection after end point
        projection_after_line_point = Point([10, 0, 0])
        shortest_line = self.line1.get_shortest_line_to_point(
            projection_after_line_point
        )
        assert isinstance(shortest_line, Line)
        self.assertEqual(shortest_line[0], self.line1[1])
        self.assertEqual(shortest_line[1], projection_after_line_point)

        # Point on ray but after endpoint
        point_on_ray_and_after_line = Point([10, 10, 0])
        shortest_line = self.line1.get_shortest_line_to_point(
            point_on_ray_and_after_line
        )
        assert isinstance(shortest_line, Line)
        self.assertEqual(shortest_line[0], self.line1[1])
        self.assertEqual(shortest_line[1], point_on_ray_and_after_line)

        # Point not on ray and projection before end point
        projection_before_line_point = Point([-10, 0, 0])
        shortest_line = self.line1.get_shortest_line_to_point(
            projection_before_line_point
        )
        assert isinstance(shortest_line, Line)
        self.assertEqual(shortest_line[0], self.line1[0])
        self.assertEqual(shortest_line[1], projection_before_line_point)

        # Point on ray but projection before end point
        point_on_ray_and_before_line = Point([-10, -10, 0])
        shortest_line = self.line1.get_shortest_line_to_point(
            point_on_ray_and_before_line
        )
        assert isinstance(shortest_line, Line)
        self.assertEqual(shortest_line[0], self.line1[0])
        self.assertEqual(shortest_line[1], point_on_ray_and_before_line)

        # Point on line null behavior
        point_on_line1 = Point([0.5, 0.5, 0])
        self.assertIsNone(
            self.line1.get_shortest_line_to_point(
                point_on_line1, point_is_on_line="null"
            )
        )

        # Point on line raise behavior
        with self.assertRaises(PointOnShapeError):
            self.line1.get_shortest_line_to_point(
                point_on_line1, point_is_on_line="raise"
            )

    def test_point_is_on_line(self):
        # Check point within bounds of line
        point_within_bounds = Point([0.5, 0.5, 0])
        self.assertTrue(self.line1.point_is_on_line(point_within_bounds))

        # Check point outside bounds of line, but on ray
        point_outside_bounds = Point([5, 5, 0])
        self.assertFalse(self.line1.point_is_on_line(point_outside_bounds))

        # Check point not on line nor ray
        point_not_on_ray = Point([1, 5, 0])
        self.assertFalse(self.line1.point_is_on_line(point_not_on_ray))

        # Check point on endpoint
        point_on_endpoint = Point(self.line1[0])
        self.assertTrue(
            self.line1.point_is_on_line(point_on_endpoint, if_on_endpoint=True)
        )
        self.assertFalse(
            self.line1.point_is_on_line(point_on_endpoint, if_on_endpoint=False)
        )


class TestLineArithmeticOperations(unittest.TestCase):
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
