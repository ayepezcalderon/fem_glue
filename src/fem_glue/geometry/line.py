import math
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
