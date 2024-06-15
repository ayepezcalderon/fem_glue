import math
import operator
from typing import Self, override
from collections.abc import  Sequence

from fem_glue.geometry.geometry import Geometry


class Point(Geometry[float]):
    """
    A point in 3D space.
    """

    @override
    def __len__(self) -> int:
        return 3

    @property
    @override
    def _ELEMENTS_TYPE(self):
        return float

    @Geometry._math_operation(operator.add)
    def __add__(self, other: float | Sequence[float]) -> Self: ...

    @Geometry._math_operation(operator.sub)
    def __sub__(self, other: float | Sequence[float]) -> Self: ...

    @Geometry._math_operation(operator.mul)
    def __mul__(self, other: float | Sequence[float]) -> Self: ...

    @Geometry._math_operation(operator.truediv)
    def __truediv__(self, other: float | Sequence[float]) -> Self: ...

    @Geometry._math_operation(operator.pow)
    def __pow__(self, other: float) -> Self: ...

    def distance(self, other: "Point") -> float:
        """
        Calculate the distance between two points.
        """
        return math.dist(self, other)

    def norm(self):
        """
        Calculate the Eucledian norm of the point.
        """
        return math.hypot(*self)

    def normalize(self):
        """
        Normalize the point into a unit vector.
        """
        return self / self.norm()
