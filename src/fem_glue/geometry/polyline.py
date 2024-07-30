from typing import override, overload
from collections.abc import Sequence

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Line, Point
from fem_glue.geometry.utils import lines_from_points, points_from_lines


class Polyline(Geometry[Line]):
    """
    A polyline with n-straight lines.
    """

    @overload
    def __init__(self, elements: Sequence[Line], /): ...

    @overload
    def __init__(self, elements: Sequence[Point], /): ...

    def __init__(self, elements: Sequence[Line | Point], /):
        if all(isinstance(e, Point) for e in elements):
            # Handle Sequence[Point] constructor
            elements = lines_from_points(elements)  # type: ignore
            self.points = list(elements)
        elif all(isinstance(e, Line) for e in elements):
            # Handle Sequence[Line] constructor
            self.points = points_from_lines(elements)  # type: ignore
        else:
            # Raise error if elements are not Lines
            raise ValueError("The elements must be either Points or Lines.")

        super().__init__(elements)  # type: ignore

    @override
    def __len__(self) -> int:
        return len(self._elements)

    def perimeter(self) -> float:
        """
        Calculate the perimeter of the polyline.
        """
        return sum(ln.length() for ln in self)
