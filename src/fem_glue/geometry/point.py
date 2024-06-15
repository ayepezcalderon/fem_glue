import math
import operator
from typing import Self
from collections.abc import Callable


class Point:
    """
    A point in 3D space.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, /):
        self._coordinates = (x, y, z)

    def __getitem__(self, index: int) -> float:
        """
        Get the coordinate at the given index.
        """
        return self._coordinates[index]

    def __iter__(self):
        return iter(self._coordinates)

    def __repr__(self):
        return f"Point({self[0]}, {self[1]}, {self[2]})"

    def __eq__(self, other: Self) -> bool:
        """
        Check if two points are equal.
        """
        if not isinstance(other, Point):
            return NotImplemented

        return self._coordinates == other._coordinates

    @staticmethod
    def _math_operation(operator: Callable, other_type: type | None = None) -> Callable:
        other_type = other_type or Point

        def wrapper(self: Self, other: Self) -> Self:
            if not isinstance(other, other_type):
                return NotImplemented

            return self.__class__(*map(operator, self, other))

        return wrapper

    __add__ = _math_operation(operator.add)
    __sub__ = _math_operation(operator.sub)
    __mul__ = _math_operation(operator.mul)
    __truediv__ = _math_operation(operator.truediv)
    __pow__ = _math_operation(operator.pow, other_type=float)

    def distance(self, other: Self) -> float:
        """
        Calculate the distance between two points.
        """
        if not isinstance(other, Point):
            raise TypeError("Expected a Point object.")

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
