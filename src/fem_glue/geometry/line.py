import math
import numpy as np
from typing import override, Self

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Point
from fem_glue._config import CONFIG


class Line(Geometry[Point]):
    """
    A straight line with 2 points.
    """

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

    def intersect(
        self, other: "Line", tol: float = 10 ** -CONFIG.precision
    ) -> "None | Point | Line":
        """
        Calculate the intersection of two lines. Return None if the lines do not
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
        """
        # Convert points to numpy arrays
        P1, P2, Q1, Q2 = [np.array(p) for p in [*self, *other]]

        # Direction vectors
        d1 = P2 - P1
        d2 = Q2 - Q1

        # Handle parallel lines
        # If cross product of direction vectors is zero, lines are parallel
        if np.linalg.norm(np.cross(d1, d2)) < tol:
            # If cross product of direction vector and vector between points is not zero,
            # lines are not collinear, so there is no intersection
            if np.linalg.norm(np.cross(Q1 - P1, d1)) >= tol:
                return None

            # Find the parametrized scalar for the P-line at the Q-line endpoints
            t1 = np.dot(Q1 - P1, d1) / np.dot(d1, d1)
            t2 = np.dot(Q2 - P1, d1) / np.dot(d1, d1)

            # Sort t-results
            t_min, t_max = sorted((t1, t2))

            # Handle non-overlapping lines
            if t_max < 0 or t_min > 1:
                return None

            # Update t-results such that they are bounded by the interval [0, 1]
            t_min_clamped = max(0, t_min)
            t_max_clamped = min(1, t_max)

            # Find intersection
            intersection_start = Point(P1 + t_min_clamped * d1)
            if np.isclose(t_min_clamped, t_max_clamped, atol=tol):
                # Start and end point are the same -> intersection is a point
                return intersection_start
            # Start and end point are different -> intersection is a segment
            intersection_end = Point(P1 + t_max_clamped * d1)
            return Line([intersection_start, intersection_end])

        # Solve for t and s
        A = np.array([d1, -d2]).T
        b = Q1 - P1

        t, s = np.linalg.lstsq(A, b, rcond=None)[0]

        # If any of the parameters are not within [0, 1], the lines do not intersect
        if not (0 <= t <= 1 and 0 <= s <= 1):
            return None

        intersection_point_L1 = P1 + t * d1
        intersection_point_L2 = Q1 + s * d2

        if not np.allclose(intersection_point_L1, intersection_point_L2, atol=tol):
            raise ValueError(
                "There was a numerical error when determining if the lines intersect."
            )

        return Point(intersection_point_L1)
