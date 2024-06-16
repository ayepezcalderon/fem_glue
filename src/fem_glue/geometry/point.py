import math
from typing import override, Self

from fem_glue.geometry.geometry import Geometry


class Point(Geometry[float]):
    """
    A point in 3D space.
    """

    @override
    def __len__(self) -> int:
        return 3

    def distance(self, other: "Point") -> float:
        """
        Calculate the distance between two points.
        """
        return math.dist(self, other)

    def norm(self) -> float:
        """
        Calculate the Eucledian norm of the point.
        """
        return math.hypot(*self)

    def normalize(self) -> Self:
        """
        Normalize the point such that the origin and the point define a unit vector.
        """
        return self / self.norm()
