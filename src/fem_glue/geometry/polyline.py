import functools

from typing import override, overload
from collections.abc import Sequence

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Line, Point
from fem_glue.geometry.utils import lines_from_points
from fem_glue._config import CONFIG


class Polyline(Geometry[Line]):
    """
    A polyline with n-straight lines.
    """

    @overload
    def __init__(
        self,
        elements: Sequence[Line],
        /,
        close: bool = False,
        strict_non_intersecting: bool = False,
    ): ...

    @overload
    def __init__(
        self,
        elements: Sequence[Point],
        /,
        close: bool = False,
        strict_non_intersecting: bool = False,
    ): ...

    def __init__(
        self,
        elements: Sequence[Line] | Sequence[Point],
        /,
        close: bool = False,
        strict_non_intersecting: bool = False,
    ):
        # Get list of lines from input
        if all(isinstance(e, Point) for e in elements):
            lines: list[Line] = lines_from_points(elements)  # type: ignore
        elif all(isinstance(e, Line) for e in elements):
            lines: list[Line] = list(elements)  # type: ignore
            # Check that lines are connected
            for i in range(len(lines) - 1):
                if lines[i][1] != lines[i + 1][0]:
                    raise ValueError(
                        f"Line '{i}' is not connected with line '{i + 1}'."
                    )
        else:
            raise TypeError("The elements must be either Points or Lines.")

        # Close the polyline if required
        is_closed = lines[-1][1] == lines[0][0]
        if not is_closed and close:
            lines.append(Line([lines[-1][1], lines[0][0]]))
            is_closed = True
        self._is_closed = is_closed

        # Raise error if intersecting but non-intersecting must be enforced
        if strict_non_intersecting and not self.is_non_intersecting():
            raise ValueError(
                "The polyline is self intersecting. "
                "If this should not raise an error, set 'strict_non_intersecting' to False."
            )

        # Set points
        self._points = [ln[0] for ln in lines]
        if not self.is_closed():
            self._points.append(lines[-1][1])

        super().__init__(lines)

    @functools.cache
    def get_self_intersections(self) -> tuple[list[Point], list[Line]]:
        points = set()
        lines = set()
        # Total iterations == (len(self) - 1) * len(self) / 2
        for ref_idx in range(len(self) - 1):
            for other in self[ref_idx + 1 :]:
                intersection = self[ref_idx].intersect(
                    other, return_mutual_endpoints=False
                )
                if isinstance(intersection, Point):
                    points.add(intersection)
                elif isinstance(intersection, Line):
                    lines.add(intersection)
                else:
                    assert intersection is None

        return sorted(points), sorted(lines)

    def is_closed(self) -> bool:
        return self._is_closed

    def is_non_intersecting(self) -> bool:
        return not any(self.get_self_intersections())

    @property
    def points(self) -> list[Point]:
        return self._points

    @override
    def __len__(self) -> int:
        return len(self._elements)

    def perimeter(self) -> float:
        """
        Calculate the perimeter of the polyline.
        """
        return round(sum(ln.length() for ln in self), CONFIG.precision)
