import math

import numpy as np

from typing import override, Self, Callable
from collections.abc import Sequence

from fem_glue.geometry._bases import SequentialGeometry
from fem_glue._config import CONFIG


class Point(SequentialGeometry[float]):
    """
    A point in 3D space.
    """

    __hash__ = SequentialGeometry.__hash__

    def __init__(self, elements: Sequence[float], /):
        elements = [float(round(i, CONFIG.precision)) for i in elements]
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

    def normalize(self) -> np.ndarray:
        """
        Normalize the point such that the origin and the point define a unit vector.
        """
        return np.array(self / self.norm())

    @override
    def _generic_operation(self, other: float | Sequence[float], op: Callable) -> Self:
        """Special function that defines how the mathematical operator "op"
        (eg. op == operator.__add__), which relates to a dunder (eg. __add__) behaves.

        For the Point, the arithmetic operation is performed on each coordinate.
        When "other" is a float, the same float acts on all coordinates.
        When a "other" is a sequence of 3 floats, each float in the sequence acts with
        its respective counterpart in the Point.

        Parameters
        ----------
        other : float | Sequence[float]
            The arithmetic operation acts on this data structure and self.
        op : Callable
            The arithmetic operation to perform.

        Returns
        -------
        Self
            The result of the arithmetic operation.
        """
        # Apply operation on each coordinate of each point of the goemetry
        if isinstance(other, Sequence):
            # Validate sequence
            if len(other) != 3:
                raise ValueError("Expected a sequence of length 3.")
            if not all(isinstance(i, float | int) for i in other):
                raise TypeError("Expected a sequence of numbers.")
        else:
            # Cast to sequence of expected length
            other = [other, other, other]

        return self.__class__([op(i, j) for i, j in zip(self, other)])

    @override
    def __contains__(self, value: float) -> bool:
        value = round(value, CONFIG.precision)

        return super().__contains__(value)

    @override
    def __eq__(self, other: "Point") -> bool:
        """
        Check if two geometries of the same type are equal.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        return all(
            math.isclose(i, j, abs_tol=CONFIG.tol)
            for i, j in zip(self, other, strict=True)
        )

    @override
    def __ne__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return not self._elements == other._elements
