from typing import override, overload
from collections.abc import Sequence

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Line, Point
from fem_glue.geometry.utils import lines_from_points, points_from_polyline_lines


class Polyline(Geometry[Line]):
    """
    A polyline with n-straight lines.
    """

    @overload
    def __init__(
        self,
        elements: Sequence[Line],
        /,
        check_is_closed: bool = False,
        check_is_non_intersecting: bool = False,
    ): ...

    def _lines_init(
        self,
        elements: Sequence[Line],
        /,
        check_is_closed: bool = False,
        check_is_non_intersecting: bool = False,
    ) -> Sequence[Line]:
        # Define as closed or not
        self._is_closed = elements[0][0] == elements[-1][1]
        if not self._is_closed:
            self._raise_not_closed_error()

        # Define as non-intersecting or not
        self._is_non_intersecting = all(
            self[ref_idx].intersect(other) is None
            for ref_idx in range(len(self) - 1)
            for other in self[ref_idx + 1 :]
        )
        if not self._is_non_intersecting:
            self._raise_not_non_intersecting_error()

        self._points = points_from_polyline_lines(elements)

        return elements

    @overload
    def __init__(
        self,
        elements: Sequence[Point],
        /,
        check_is_closed: bool = False,
        check_is_non_intersecting: bool = False,
    ): ...

    def _points_init(
        self,
        elements: Sequence[Point],
        /,
        check_is_closed: bool = False,
        check_is_non_intersecting: bool = False,
    ) -> Sequence[Line]:
        self._points = list(elements)

        return lines_from_points(elements)

    def __init__(
        self,
        elements: Sequence[Line | Point],
        /,
        check_is_closed: bool = False,
        check_is_non_intersecting: bool = False,
    ):
        if all(isinstance(e, Point) for e in elements):
            _elements = self._points_init(
                elements, check_is_closed, check_is_non_intersecting
            )
        elif all(isinstance(e, Line) for e in elements):
            _elements = self._lines_init(
                elements, check_is_closed, check_is_non_intersecting
            )
        else:
            # Raise error if elements are not Lines
            raise ValueError("The elements must be either Points or Lines.")

        super().__init__(_elements)

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
