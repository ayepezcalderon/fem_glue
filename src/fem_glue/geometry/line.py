import math
import numpy as np
from typing import override, Self

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Point


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
        return math.dist(*self)

    def normalize(self) -> Self:
        """
        Normalize the line such that its ends define a unit vector.
        """
        return self / self.length()

    def intersect(self, other: "Line", tol: float = 1e-9) -> "None | Point | Line":
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
                    - When L1(t=t0) = Q1 and L2(t=t1) = Q2, then:
                        - t1, t2 in interval [0, 1]
                        - t1 != t2
                    - The endpoints of the intersection segment are defined by L1(t) at:
                        max(0, min(t1, t2))
                        min(1, max(t1, t2))

                - If a point intersection:
                    - Option 1:
                        - Same situation as for segment intersection, but t1 == t2 and
                        both endpoints definitions are equivalent

                    - Option 2:
                        - 


        """
        # Convert points to numpy arrays
        (P1, P2, Q1, Q2) = [np.array(p) for p in [*self, *other]]

        # Direction vectors
        d1 = P2 - P1
        d2 = Q2 - Q1

        # Handle parallel lines
        # If cross product of direction vectors is zero, lines are parallel
        if np.linalg.norm(np.cross(d1, d2)) < tol:
            # If cross product of direction vector and vector between points is zero,
            # lines are collinear
            if np.linalg.norm(np.cross(Q1 - P1, d1)) < tol:
                # Find the parametrized scalar for the P-line at the Q-line endpoints
                t0 = np.dot(Q1 - P1, d1) / np.dot(d1, d1)
                t1 = np.dot(Q2 - P1, d1) / np.dot(d1, d1)
                
                # Sort t-results
                t_min, t_max = sorted((t0, t1))

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

            else:
                # Lines are parallel but not collinear
                return None

        # Solve for t and s
        A = np.array([d1, -d2]).T
        b = Q1 - P1

        t, s = np.linalg.lstsq(A, b, rcond=None)[0]

        intersection_point_L1 = P1 + t * d1
        intersection_point_L2 = Q1 + s * d2

        if np.allclose(intersection_point_L1, intersection_point_L2, atol=tol):
            return "Lines intersect at a single point", intersection_point_L1.tolist()
        else:
            return "Lines do not intersect"
