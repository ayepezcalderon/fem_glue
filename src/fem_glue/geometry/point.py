import math
from typing import override, Self
from collections.abc import Iterable

from fem_glue.geometry.geometry import Geometry
from fem_glue._config import CONFIG


class Point(Geometry[float]):
    """
    A point in 3D space.
    """

    def __init__(self, elements: Iterable[float], /):
        elements = [round(i, CONFIG.precision) for i in elements]
        super().__init__(elements)

    @override
    def __len__(self) -> int:
        return 3

    def distance(self, other: "Point") -> float:
        """
        Calculate the distance between two points.
        """
        return round(math.dist(self, other), CONFIG.precision)

    def norm(self) -> float:
        """
        Calculate the Eucledian norm of the point.
        """
        return round(math.hypot(*self), CONFIG.precision)

    def normalize(self) -> Self:
        """
        Normalize the point such that the origin and the point define a unit vector.
        """
        return self / self.norm()
