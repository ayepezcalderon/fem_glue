"""Defines the behavior of the Line class.

The Line defines straight lines in 3D space.
"""

import functools
import math
from collections.abc import Sequence
from typing import Literal, Self, assert_never, override

import numpy as np

from fem_glue._config import CONFIG
from fem_glue._utils import check_literal, tol_compare
from fem_glue.geometry import Point
from fem_glue.geometry._bases import SequentialGeometry
from fem_glue.geometry._exceptions import PointNotOnShapeError, PointOnShapeError

type SelfRaise = Literal["self", "raise"]
type NullRaise = Literal["null", "raise"]


class Line(SequentialGeometry[Point]):
    """A straight line defined by 2 points in 3D space."""

    def __init__(self, elements: Sequence[Point], /):
        """Define a straight line in 3D space.

        Parameters
        ----------
        elements : Sequence[Point]
            Sequence of 2 Points, each of which is associated to an endpoint of the
            line. The first Point is the start endpoint and the second Point is the
            end endpoint. This convention defines the direction of the line.

        """
        # Check that the points in the line are not the same
        if elements[0] == elements[1]:
            raise ValueError("A line cannot be constructed from two identical points.")

        super().__init__(elements)

    @override
    def __len__(self) -> int:
        return 2

    def length(self) -> float:
        """Calculate the length of the line."""
        return round(math.dist(*self), CONFIG.precision)

    def normalize(self) -> Self:
        """Normalize the line such that its ends define a unit line."""
        return self / self.length()

    def as_vector(self) -> np.ndarray:
        """Return the line as a vector centered at the origin."""
        return np.array(self[1] - self[0])

    def dir_unit_vector(self) -> np.ndarray:
        """Calculate the unit direction vector of the line."""
        return self.as_vector() / self.length()

    def get_point_projection_on_ray(
        self,
        point: Point,
        point_is_on_ray: SelfRaise = "self",
    ) -> Point:
        """Calculate the projection of the given point onto the ray of the line.

        Parameters
        ----------
        point : Point
            The point to project onto the ray of the line.
        point_is_on_ray : SelfRaise
            Specifies the behavior when the given point is on the ray of the line.
            If "self", the given point is returned.
            If "raise", an error is raised.
            Default is "self".

        Returns
        -------
        Point
            The projection of the given point onto the ray of the line.

        """
        check_literal("point_is_on_ray", point_is_on_ray, SelfRaise)

        # Vector between the start point of the line and the given point
        line_to_point_vector = point - self[0]

        # Projection vector of the vector above onto the ray of the line
        projection_vector = (
            np.dot(line_to_point_vector.as_array(), self.dir_unit_vector())
            * self.dir_unit_vector()
        )

        # Given point projected onto the ray of the line
        projected_point = self[0] + list(projection_vector)

        # Handle case where the point is on the ray of the line
        if np.allclose(point.as_array(), projected_point.as_array(), atol=CONFIG.tol):
            if point_is_on_ray == "raise":
                raise PointOnShapeError("The point is on the ray of the line.")
            elif point_is_on_ray == "self":
                return point
            else:
                assert_never(point_is_on_ray)  # pragma: no cover

        # Rounding to 1 less decimal than the config seems to behave better
        return projected_point.round(CONFIG.precision - 1)

    def get_point_position_on_ray(
        self,
        point: Point,
        normalized: bool = True,
        point_is_not_on_ray: NullRaise = "null",
    ) -> float | None:
        """Calculate the position of the point on the line's ray coordinate system.

        This 1D coordinate system is defined on the direction of the line and has the
        origin on the start point of the line.
        The coordinate system is also normalized such that its unit length is
        equal to the length of the line, unless otherwise specified.

        Parameters
        ----------
        point : Point
            The point to check.
        normalized : bool
            Specifies wether the coordinate system is normalized such that its unit
            length is equal to the length of the line.
            Default is True.
        point_is_not_on_ray : NullRaise
            Defines the behavior when the given point is not on the ray of the line.
            If "null", None is returned.
            If "raise", an error is raised.
            Default is "null".

        Returns
        -------
        float | None
            The position of the point on the coordinate system.

        """
        check_literal("point_is_on_ray", point_is_not_on_ray, NullRaise)

        # Handle case where the point is not on the ray of the line
        # Unsigned direction vectors of line and line between point and point in line
        # must be equal
        test_dir_vector = Line([self[0], point]).dir_unit_vector()
        if not any(
            np.allclose(
                self.dir_unit_vector(),
                sign * test_dir_vector,
                atol=CONFIG.tol,
            )
            for sign in [1, -1]
        ):
            if point_is_not_on_ray == "raise":
                raise PointNotOnShapeError("The point is not on the ray of the line.")
            elif point_is_not_on_ray == "null":
                return None
            else:
                assert_never(point_is_not_on_ray)  # pragma: no cover

        position = np.dot(np.array(point - self[0]), self.dir_unit_vector())

        return round(
            position / self.length() if normalized else position,
            CONFIG.precision,
        )

    def get_point_projection_on_line(
        self,
        point: Point,
        point_is_on_line: SelfRaise = "self",
        projection_is_not_on_line: NullRaise = "null",
    ) -> Point | None:
        """Calculate the projection of the given point onto the line.

        Parameters
        ----------
        point : Point
            The point to project onto the line.
        point_is_on_line : SelfRaise
            Specifies the behavior when the given point is on the line.
            If "self", the given point is returned.
            If "raise", an error is raised.
            Default is "self".
        projection_is_not_on_line : NullRaise
            Specifies the behavior when the projection of the point is not on the line.
            If "null", None is returned.
            If "raise", an error is raised.
            Default is "null".

        Returns
        -------
        Point | None
            The projection of the given point onto the line.

        """
        check_literal("point_is_on_line", point_is_on_line, SelfRaise)
        check_literal("projection_is_not_on_line", projection_is_not_on_line, NullRaise)

        point_projection_on_ray = self.get_point_projection_on_ray(
            point, point_is_on_ray="self"
        )

        projection_pos_on_ray = self.get_point_position_on_ray(
            point_projection_on_ray, normalized=True, point_is_not_on_ray="null"
        )
        # Finding position of projection, so it must be on the ray
        assert projection_pos_on_ray is not None

        # Handle case where the projection is not on the line
        if tol_compare(projection_pos_on_ray, 0, op="le") or tol_compare(
            projection_pos_on_ray, 1, op="ge"
        ):
            if projection_is_not_on_line == "raise":
                raise PointNotOnShapeError(
                    "The projection of the point is not on the line."
                )
            elif projection_is_not_on_line == "null":
                return None
            else:
                assert_never(projection_is_not_on_line)  # pragma: no cover

        # If projection is on line and is equal to the point, point is on line
        if point_projection_on_ray == point:
            if point_is_on_line == "raise":
                raise PointOnShapeError("The point is on the line.")
            elif point_is_on_line == "self":
                return point
            else:
                assert_never(point_is_on_line)  # pragma: no cover

        return point_projection_on_ray

    def get_shortest_line_to_point(
        self,
        point: Point,
        point_is_on_line: NullRaise = "null",
    ) -> Self | None:
        """Calculate the shortest line between the line and the given point.

        Start point is point on line, end point is given point.

        Parameters
        ----------
        point : Point
            The point to which the shortest line is calculated.
        point_is_on_line : NullRaise
            Specifies the behavior when the given point is on the line.
            If "null", None is returned.
            If "raise", an error is raised.
            Default is "null".

        Returns
        -------
        Self | None
            The shortest line between the line and the point.

        """
        # Find the projection of the point onto the line
        # If the point is on the line, handle that case
        point_projection_on_line = self.get_point_projection_on_line(
            point,
            point_is_on_line="raise" if point_is_on_line == "raise" else "self",
            projection_is_not_on_line="null",
        )
        if point_projection_on_line == point:
            return None

        # If the projection of the point is not on the line, the endpoint of the line
        # closest to the point defines the shortest line to the point
        if point_projection_on_line is None:
            if point.distance(self[0]) < point.distance(self[1]):
                return self.__class__([self[0], point])
            return self.__class__([self[1], point])

        # If the projection of the point is on the line, the projection defines
        # the shortest line to the point
        return self.__class__([point_projection_on_line, point])

    def point_is_on_line(self, point: Point, if_on_endpoint: bool = False) -> bool:
        """Check if a point is on the line.

        Parameters
        ----------
        point : Point
            The point to check.
        if_on_endpoint : bool
            Specifies the return value if the point is on an endpoint of the line.

        Returns
        -------
        bool
            True if the point is on the line, False otherwise.

        """
        # If the point is on an endpoint, return according to the specified protocol
        if point in self:
            return if_on_endpoint

        return point == self.get_point_projection_on_line(
            point, point_is_on_line="self", projection_is_not_on_line="null"
        )

    def is_parallel(self, other: "Line", tol: float = CONFIG.tol) -> bool:
        """Check if two lines are parallel.

        Parameters
        ----------
        other : "Line"
            The line to compare against.
        tol : float
            Tolerance specifying how close lines have to be together to be considered
            parallel.
            Usually a very small number for stabilizing floating point operations.

        Returns
        -------
        bool
            True if the lines are parallel, False otherwise.

        """
        self._check_other_is_line(other)

        # If cross product of direction vectors is zero, lines are parallel
        are_parallel = (
            np.linalg.norm(np.cross(self.as_vector(), other.as_vector())) < tol
        )

        assert isinstance(are_parallel, np.bool_)

        return bool(are_parallel)

    def is_collinear(self, other: "Line", tol: float = CONFIG.tol) -> bool:
        """Check if two lines are collinear.

        Parameters
        ----------
        other : "Line"
            The line to compare against.
        tol : float
            Tolerance specifying how close lines have to be together to be considered
            collinear.
            Usually a very small number for stabilizing floating point operations.

        Returns
        -------
        bool
            True if the lines are collinear, False otherwise.

        """
        self._check_other_is_line(other)

        v1 = self.__class__(sorted(self))
        v2 = other.__class__(sorted(other))

        if v1 == v2:
            return True

        # The interpolation vector is the difference between different points
        interpoint_vector = v2[0] - v1[0] if v2[0] != v1[0] else v2[1] - v1[1]

        # If cross product of direction vector and vector between points is not zero,
        # lines are not collinear
        are_collinear = (
            np.linalg.norm(np.cross(v1.as_vector(), interpoint_vector.as_array())) < tol
        )

        assert isinstance(are_collinear, np.bool_)

        return bool(are_collinear)

    @functools.cache
    def intersect(
        self,
        other: "Line",
        return_mutual_endpoints: bool = True,
        tol: float = CONFIG.tol,
    ) -> "None | Point | Line":
        """Calculate the intersection between two lines.

        Return None if the lines do not intersect, a Point if they intersect at a
        single point, and a Line if they intersect at a segment.

        Calculation approach:
            - Parametrize each the lines as:
                L1(t) = P1 + t * d1
                L2(s) = Q1 + s * d2
                - where the direction vectors are:
                    d1 = P2 - P1
                    d2 = Q2 - Q1
                - and where the lines' spans coincide with the scalar parameters such
                  that:
                    L1 span -> t in interval [0, 1]
                    L2 span -> s in interval [0, 1]

            - Solve for the intersection:
                - If a segment intersection:
                    - Lines are parallel and collinear
                    - When L1(t=t1) = Q1 and L1(t=t2) = Q2, then:
                        - at least one of t1, t2 in interval [0, 1]
                    - The endpoints of the intersection segment are defined by L1(t) at:
                        t_min = max(0, min(t1, t2))
                        t_max = min(1, max(t1, t2))
                    - t_min != t_max

                - If a point intersection:
                    - Option 1:
                        - Same situation as for segment intersection, but t_min == t_max
                        and both endpoints definitions are equivalent

                    - Option 2:
                        - L1(t) = L2(s) and t, s are within the interval [0, 1]
                            - t, s can be obtained from the system of linear equations
                            defined by L1(t) = L2(s)

        Parameters
        ----------
        other : "Line"
            The against which intersections are computed.
        return_mutual_endpoints : bool
            Determines wether to consider mutual endpoints between the lines as
            intersections. By default True.
        tol : float
            Tolerance specifying how close lines have to be together to intersect.
            Usually a very small number for stabilizing floating point operations.

        Returns
        -------
        "None | Point | Line"
            Intersection between the lines. None if they don't intersect,
            a Point if they intersect at a point, and a Line if they intersect at a
            segment (ie. infinitely many points).

        """
        # Convert points to numpy arrays
        P1, P2, Q1, Q2 = (np.array(p) for p in [*self, *other])

        # Direction vectors
        d1 = P2 - P1
        d2 = Q2 - Q1

        # Handle parallel lines
        if self.is_parallel(other, tol=tol):
            # If lines are parallel but not collinear, there is no intersection
            if not self.is_collinear(other, tol=tol):
                return None

            # Find the parametrized scalar for the P-line at the Q-line endpoints
            t1 = np.dot(Q1 - P1, d1) / np.dot(d1, d1)
            t2 = np.dot(Q2 - P1, d1) / np.dot(d1, d1)

            # Sort t-results
            t_min, t_max = sorted((t1, t2))

            # Handle non-overlapping lines
            if tol_compare(t_max, 0, op="lt") or tol_compare(t_min, 1, op="gt"):
                return None

            # Update t-results such that they are bounded by the interval [0, 1]
            t_min_clamped = max(0, t_min)
            t_max_clamped = min(1, t_max)

            # Find intersection
            intersection_start = Point(P1 + t_min_clamped * d1)  # type: ignore
            if np.isclose(t_min_clamped, t_max_clamped, atol=tol):
                # Start and end point are the same -> intersection is a point
                return intersection_start if return_mutual_endpoints else None
            # Start and end point are different -> intersection is a segment
            intersection_end = Point(P1 + t_max_clamped * d1)  # type: ignore
            return Line([intersection_start, intersection_end])

        # Check if lines share a point and return appropriate value
        for _point1 in self:
            for _point2 in other:
                if _point1 == _point2:
                    return Point(_point1) if return_mutual_endpoints else None

        # Solve for t and s
        A = np.array([d1, -d2]).T
        b = Q1 - P1

        t, s = np.linalg.lstsq(A, b, rcond=None)[0]

        # If any of the parameters are not within (0, 1), the lines do not intersect
        if not (0 < t < 1 and 0 < s < 1):
            return None

        intersection_point_L1 = P1 + t * d1
        intersection_point_L2 = Q1 + s * d2

        if not np.allclose(intersection_point_L1, intersection_point_L2, atol=tol):
            raise ValueError(
                "There was a numerical error when determining if the lines intersect."
            )  # pragma: no cover

        return Point(intersection_point_L1)

    @staticmethod
    def _check_other_is_line(other):
        if not isinstance(other, Line):
            raise TypeError("The other object must be a Line.")
