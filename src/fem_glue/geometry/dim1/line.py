import functools
import math
import numpy as np

from typing import override, Self, Literal
from collections.abc import Sequence

from fem_glue.geometry._bases import SequentialGeometry
from fem_glue.geometry import Point
from fem_glue._config import CONFIG
from fem_glue._utils import tol_compare


class Line(SequentialGeometry[Point]):
    """
    A straight line with 2 points.
    """

    def __init__(self, elements: Sequence[Point], /):
        # Check that the points in the line are not the same
        if elements[0] == elements[1]:
            raise ValueError("A line cannot be constructed from two identical points.")

        super().__init__(elements)

    @override
    def __len__(self) -> int:
        return 2

    def length(self) -> float:
        """
        Calculate the length of the line.
        """
        return round(math.dist(*self), CONFIG.precision)

    def normalize(self) -> Self:
        """
        Normalize the line such that its ends define a unit vector.
        """
        return self / self.length()

    def as_vector(self) -> np.ndarray:
        """
        Return the line as a vector centered at the origin.
        """
        return np.array(self[1] - self[0])

    def dir_unit_vector(self) -> np.ndarray:
        """
        Calculate the unit direction vector of the line.
        """
        return self.as_vector() / self.length()

    def point_projection_on_ray(
        self,
        point: Point,
        point_is_on_ray: Literal["self", "raise"] = "self",
    ) -> Point:
        """
        Calculate the projection of the given point onto the ray of the line.

        Parameters
        ----------
        point : Point
            The point to project onto the ray of the line.
        point_is_on_ray : Literal["self", "raise"]
            Specifies the behavior when the given point is on the ray of the line.
            If "self", the given point is returned.
            If "raise", an error is raised.
            Default is "null".

        Returns
        -------
        Point
            The projection of the given point onto the ray of the line.
        """
        # Vector between the start point of the line and the given point
        line_to_point_vector = point - self[0]

        # Projection vector of the vector above onto the ray of the line
        projection_vector = (
            np.dot(line_to_point_vector, self.dir_unit_vector())
            * self.dir_unit_vector()
        )

        # Given point projected onto the ray of the line
        projected_point = self[0] + projection_vector

        # Check if the point is on the ray of the line
        # If it is, handle the situation as specified
        if np.allclose(np.array(point), projected_point):
            if point_is_on_ray == "self":
                return point
            if point_is_on_ray == "raise":
                raise ValueError("The point is on the ray of the line.")

        return projected_point

    def point_position_on_ray(
        self,
        point: Point,
        normalized: bool = True,
        not_on_ray: Literal["null", "raise", "project"] = "null",
    ) -> float | None:
        """
        Calculate the position of the point on the coordinate system of the ray of the line.
        This 1D coordinate system is defined on the direction of the line and has the
        origin on the start point of the line.
        The coordinate system is also normalized such that its unit length is
        equal to the length of the line unless otherwise specified.

        Parameters
        ----------
        point : Point
            The point to check.
        normalized : bool
            Specifies wether the coordinate system is normalized such that its unit
            length is equal to the length of the line.
            Default is True.
        not_on_ray : Literal["null", "raise", "project"]
            Defines the behavior when the given point is not on the ray of the line.
            If "null", None is returned.
            If "raise", an error is raised.
            If "project", the position of the point is projected onto the ray of the
            line, and the position of the projection is returned.
            Default is "null".

        Returns
        -------
        float | None
            The position of the point (or its projection) on the
            (potentially normalized) coordinate system.
            If not_on_ray is "null" and the point is not on the ray of the line, None.
        """
        # Check if the point is on the ray of the line
        # If it is not, handle the situation as specified
        new_line = Line([self[0], point])
        if not np.allclose(
            self.dir_unit_vector(), new_line.dir_unit_vector(), atol=CONFIG.tol
        ):
            if not_on_ray == "null":
                return None
            if not_on_ray == "raise":
                raise ValueError("The point is not on the ray of the line.")
            if not_on_ray == "project":
                point = self.point_projection_on_ray(point, point_is_on_ray="raise")

        normalized_position = np.dot(point - self[0], self.dir_unit_vector())

        return (
            normalized_position if normalized else normalized_position * self.length()
        )

    def contains_point(self, point: Point, count_endpoints: bool = True) -> bool:
        """
        Check if the line contains the given point.

        Parameters
        ----------
        point : Point
            The point to check.
        count_endpoints : bool
            If the point coincides with an endpoint of the line, specifies whether
            the line is considered to contain the point.

        Returns
        -------
        bool
            True if the line contains the point, False otherwise.
        """
        if point in self:
            return count_endpoints

        pos_on_ray = self.point_position_on_ray(
            point, normalized=True, not_on_ray="null"
        )

        # Normal comparison is possible since "in" operation checked for tolerance
        return pos_on_ray is not None and 0 < pos_on_ray < 1

    def shortest_line_to_point(
        self,
        point: Point,
        point_is_on_line: Literal["null", "raise"] = "null",
    ) -> Self | None:
        """
        Calculate the shortest line between the line and the given point.
        Start point is point on line, end point is given point.

        TODO: make the implementation more efficient (less repeated operations)

        Parameters
        ----------
        point : Point
            The point to which the shortest line is calculated.
        point_is_on_line : Literal["null", "raise"]
            Specifies the behavior when the given point is on the line.
            If "null", None is returned.
            If "raise", an error is raised.
            Default is "null".

        Returns
        -------
        Self | None
            The shortest line between the line and the point.
        """
        # Handle case where point is on line
        if self.contains_point(point, count_endpoints=True):
            if point_is_on_line == "null":
                return None
            if point_is_on_line == "raise":
                raise ValueError("The point is on the line.")

        pos_on_ray = self.point_position_on_ray(
            point, normalized=True, not_on_ray="project"
        )
        assert isinstance(pos_on_ray, float)

        # If the projection of the point onto the ray is outside the line,
        # then the point on the line is the closest endpoint
        if tol_compare(pos_on_ray, 0, op="le"):
            return self.__class__([self[0], point])
        if tol_compare(pos_on_ray, 1, op="ge"):
            return self.__class__([self[1], point])

        # If the projection of the point is inside the line,
        # then the point on the line is the projection
        point_on_line = self.point_projection_on_ray(point, point_is_on_ray="raise")

        return self.__class__([point_on_line, point])

    def is_parallel(self, other: "Line", tol: float = CONFIG.tol) -> bool:
        """
        Check if two lines are parallel.

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
        """
        Check if two lines are collinear.

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
            np.linalg.norm(np.cross(v1.as_vector(), interpoint_vector)) < tol
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
        """
        Calculate the intersection between two lines. Return None if the lines do not
        intersect, a point if they intersect at a single point, and a line if
        they intersect at a segment.

        Calculation approach:
            - Parametrize each the lines as:
                L1(t) = P1 + t * d1
                L2(s) = Q1 + s * d2
                - where the direction vectors are:
                    d1 = P2 - P1
                    d2 = Q2 - Q1
                - and where the lines' spans coincide with the scalar parameters such that:
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
        P1, P2, Q1, Q2 = [np.array(p) for p in [*self, *other]]

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
            )

        return Point(intersection_point_L1)

    @staticmethod
    def _check_other_is_line(other):
        if not isinstance(other, Line):
            raise TypeError("The other object must be a Line.")
