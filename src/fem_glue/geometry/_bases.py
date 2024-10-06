import functools
import operator

from typing import Self, overload, override
from collections.abc import Callable, Sequence, Iterator
from abc import abstractmethod


class SequentialGeometry[T](Sequence[T]):
    """
    Abstract base class for all geometries.
    """

    def __init__(self, elements: Sequence[T], /):
        self._elements = tuple(elements)

        if len(self._elements) != len(self):
            raise ValueError(
                f"The number of elements in the iterabale must be equal to {len(self)}."
            )

    @overload
    def __getitem__(self, index: int) -> T:
        """
        Get the element at the given index.
        """

    @overload
    def __getitem__(self, index: slice) -> list[T]:
        """
        Get the elements for the given range in a list.
        """

    @override
    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, int):
            return self._elements[index]
        return list(self._elements[index])

    @abstractmethod
    @override
    def __len__(self) -> int: ...

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._elements)

    @override
    def __reversed__(self) -> Iterator[T]:
        return reversed(self._elements)

    @override
    def index(self, value: T, start: int = 0, stop: int = 9223372036854775807) -> int:
        """
        Get the index of the given element.
        """
        return self._elements.index(value, start, stop)

    @override
    def count(self, value: float) -> int:
        """
        Count the occurrences of the given element.
        """
        return self._elements.count(value)

    def reversed(self) -> Self:
        """
        Return a reversed version of the geometry.
        """
        return self.__class__(tuple(reversed(self)))

    @override
    def __repr__(self):
        return f"{self.__class__}([{', '.join(str(i) for i in self)}])"

    @override
    def __eq__(self, other: Self) -> bool:
        """
        Check if two geometries of the same type are equal.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements == other._elements

    @override
    def __ne__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements != other._elements

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements < other._elements

    def __gt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements > other._elements

    def __le__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements <= other._elements

    def __ge__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements >= other._elements

    @override
    def __hash__(self) -> int:
        return hash(self._elements)

    def _generic_operation(
        self,
        other: float | Sequence[float],
        op: Callable[[T, float | Sequence[float]], Self],
    ) -> Self:
        """Special function that defines how the mathematical operator "op"
        (eg. op == operator.__add__), which relates to a dunder (eg. __add__) behaves.

        The default generic operation defined here performs "op" between "other" and
        each element of self.

        Subclasses may override this function to override the default generic
        operation behavior defined here.

        Parameters
        ----------
        other : float | Sequence[float]
            The arithmetic operation acts on this data structure and self.
        op : Callable[T, float | Sequence[float]]
            The arithmetic operation to perform.

        Returns
        -------
        Self
            The result of the arithmetic operation.
        """

        return self.__class__([op(i, other) for i in self])

    @staticmethod
    def _math_operation(f: Callable) -> Callable:
        # Find operator from function name
        op = getattr(operator, f.__name__.strip("__"))

        @functools.wraps(f)
        def wrapper(self: Self, other: float | Sequence[float]) -> Self:
            if not isinstance(other, float | int | Sequence):
                return NotImplemented

            return self._generic_operation(other, op)

        return wrapper

    @_math_operation
    def __add__(self, other: float | Sequence[float]) -> Self: ...

    @_math_operation
    def __sub__(self, other: float | Sequence[float]) -> Self: ...

    @_math_operation
    def __mul__(self, other: float | Sequence[float]) -> Self: ...

    @_math_operation
    def __truediv__(self, other: float | Sequence[float]) -> Self: ...

    @_math_operation
    def __floordiv__(self, other: float | Sequence[float]) -> Self: ...

    def __pow__(self, other: float) -> Self:
        if not isinstance(other, float | int):
            return NotImplemented

        return self.__class__([operator.pow(i, other) for i in self])
