from typing import Never, override, overload
from collections.abc import Sequence

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Line, Point
from fem_glue.geometry.utils import lines_from_points, points_from_polyline_lines


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
            self._points = list(elements)
            elements = lines_from_points(elements)  # type: ignore
        elif all(isinstance(e, Line) for e in elements):
            # Handle Sequence[Line] constructor
            self._points = points_from_polyline_lines(elements)  # type: ignore
        else:
            # Raise error if elements are not Lines
            raise ValueError("The elements must be either Points or Lines.")

        super().__init__(elements)  # type: ignore

    def get_points(self) -> list[Point]:
        return self._points

    @override
    def __len__(self) -> int:
        return len(self._elements)

    def perimeter(self) -> float:
        """
        Calculate the perimeter of the polyline.
        """
        return sum(ln.length() for ln in self)
