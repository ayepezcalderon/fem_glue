"""Defines the behavior of the Point class.

The Point defines points in 3D space.
"""

import math
from collections.abc import Callable, Sequence
from typing import Self, override

import numpy as np

from fem_glue._config import CONFIG
from fem_glue.geometry._bases import SequentialGeometry


class Point(SequentialGeometry[float]):
    """A point in 3D space."""

    __hash__ = SequentialGeometry.__hash__

    def __init__(self, elements: Sequence[float], /):
        """Define a point in 3D space.

        Parameters
        ----------
        elements : Sequence[float]
            Sequence of 3 numbers, each of which is associated to a dimension in
            3D space.

        """
        elements = [float(round(i, CONFIG.precision)) for i in elements]
        super().__init__(elements)

    @override
    def __len__(self) -> int:
        return 3

    def distance(self, other: "Point") -> float:
        """Calculate the distance between two points."""
        return round(math.dist(self, other), CONFIG.precision)

    def norm(self) -> float:
        """Calculate the Eucledian norm of the point."""
        return round(math.hypot(*self), CONFIG.precision)

    def normalize(self) -> np.ndarray:
        """Normalize the point so that the origin and the point define a unit vector."""
        return np.array(self / self.norm())

    def as_array(self) -> np.ndarray:
        """Convert the point to a numpy array."""
        return np.array(self)

    def round(self, precision: int = CONFIG.precision) -> Self:
        """Round the entries of the point to a specific precision.

        Parameters
        ----------
        precision : int
            The precision to round to.
            Default is the precision set in the configuration.

        Returns:
        -------
        Self
            The point with rounded entries.

        """
        return self.__class__([round(i, precision) for i in self])

    @override
    def _generic_operation(
        self,
        other: float | Sequence[float],
        op: Callable[[float, float], float],
    ) -> Self:
        """Define how arithmetic operation dunders should behave (eg. __add__).

        Special function that defines how the mathematical operator "op"
        (eg. op == operator.add) relates to a dunder (eg. __add__) behaves.

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

        Returns:
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

        return self.__class__([op(i, j) for i, j in zip(self, other, strict=True)])

    @override
    def __contains__(self, value: float) -> bool:
        value = round(value, CONFIG.precision)

        return super().__contains__(value)

    @override
    def __eq__(self, other: "Point") -> bool:
        """Check if two geometries of the same type are equal."""
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
