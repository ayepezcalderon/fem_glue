import math
import operator
import functools
from typing import Self, get_type_hints
from collections.abc import Callable, Sequence


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

    def __len__(self) -> int:
        return 3

    def __iter__(self):
        return iter(self._coordinates)

    def __reversed__(self):
        return reversed(self._coordinates)

    def index(self, value: float) -> int:
        """
        Get the index of the given value.
        """
        return self._coordinates.index(value)

    def count(self, value: float) -> int:
        """
        Count the occurrences of the given value.
        """
        return self._coordinates.count(value)

    def __repr__(self):
        return f"Point({self[0]}, {self[1]}, {self[2]})"

    def __eq__(self, other: "Point") -> bool:
        """
        Check if two points are equal.
        """
        if not isinstance(other, Point):
            return NotImplemented

        return self._coordinates == other._coordinates

    @staticmethod
    def _math_operation(operator: Callable) -> Callable:
        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(self, other) -> Self:
                if not isinstance(other, get_type_hints(f)["other"]):
                    return NotImplemented

                if isinstance(other, Sequence):
                    if len(other) != 3:
                        raise ValueError("Expected an iterable of length 3.")
                    if not all(isinstance(i, (int, float)) for i in other):
                        raise TypeError("Expected an iterable of numbers.")

                if isinstance(other, (int, float)):
                    other = (other, other, other)

                return self.__class__(*map(operator, self, other))

            return wrapper

        return decorator

    @_math_operation(operator.add)
    def __add__(self, other: float | Sequence) -> Self: ...

    @_math_operation(operator.sub)
    def __sub__(self, other: float | Sequence) -> Self: ...

    @_math_operation(operator.mul)
    def __mul__(self, other: float | Sequence) -> Self: ...

    @_math_operation(operator.truediv)
    def __truediv__(self, other: float | Sequence) -> Self: ...

    @_math_operation(operator.pow)
    def __pow__(self, other: float) -> Self: ...

    def distance(self, other: "Point") -> float:
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
