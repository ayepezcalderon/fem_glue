import functools
import operator
from typing import Self, overload, override
from collections.abc import Callable, Iterable, Iterator, Sequence
from abc import abstractmethod


class Geometry[T](Sequence[T]):
    """
    Abstract base class for all geometries.
    """

    def __init__(self, elements: Iterable[T], /):
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

    @override
    def __repr__(self):
        return f"{self.__class__}([{', '.join(str(i) for i in self)}])"

    @override
    def __eq__(self, other: object) -> bool:
        """
        Check if two geometries of the same type are equal.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._elements == other._elements

    @staticmethod
    def _math_operation(f: Callable) -> Callable:
        # Find operator from function name
        op = getattr(operator, f.__name__.strip("__"))

        @functools.wraps(f)
        def wrapper(self: Self, other: float | Sequence[float]) -> Self:
            if not isinstance(other, float | Sequence):
                return NotImplemented

            if isinstance(other, Sequence):
                # Validate sequence
                if len(other) != len(self):
                    raise ValueError(f"Expected an iterable of length {len(self)}.")
                if not all(isinstance(i, float) for i in other):
                    raise TypeError("Expected an iterable of numbers.")
            else:
                # Cast to sequence of expected length
                other = [other] * len(self)

            return self.__class__(map(op, self, other))

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
        if not isinstance(other, float):
            return NotImplemented

        return self.__class__(operator.pow(i, other) for i in self)
