"""Defines the behavior of the Polyline class.

The Polyline defines a sequence of n-connected straight lines in 3D space.
"""

import functools
from collections.abc import Sequence
from typing import overload, override

from fem_glue._config import CONFIG
from fem_glue.geometry import Point
from fem_glue.geometry._bases import SequentialGeometry
from fem_glue.geometry.dim1 import Line
from fem_glue.geometry.utils import lines_from_points


class Polyline(SequentialGeometry[Line]):
    """A polyline defined by n-straight lines connected together in 3D space.

    The only hard constraint of a polyline is that lines must be connected to each
    other. That is, the end point of line n must have the same coordinates as the
    start point of line n+1.

    The following optional constraints can also be enforced:
        - closed: The end point of the last line must have the same coordinates
        as the start point of the first line.
        - non-intersecting: The lines of the polyline cannot intersect each other
        anywhere.

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
        """Define a polyline in 3D space.

        The Polyline defines a sequence of n-connected Lines in 3D space.

        Parameters
        ----------
        elements : Sequence[Line] | Sequence[Point]
            Sequence of lines or sequence of points.
            If a sequence of lines, the lines have to be connected to each other.
            This means that the end point of line n must have the same coordinates as
            the start point of line n+1.
            If a sequence of points, the points define the lines in the polyline.
            That is, point n defines the end point of line n-1 and the start point
            of line n. The exception to this rule is when the point is the first or
            last point of the sequence. For the first point the start point only
            defines the the start point of the first line, and for the last point the
            end point only defines the end point of the last line. These first/last
            special cases have exceptions when `close=True`.
        close : bool
            If True, closes the polyline.
            If a sequence of lines was passed and the last line is not connected to the
            first, a new last line is instroduced connecting the former last line and
            the first line.
            If a sequence of points was passed and the first and last points are not
            equal, a new last line is instroduced connecting these points.
            Default is False.
        strict_non_intersecting : bool
            If True, the lines of the polyline cannot intersect each other anywhere.
            If True and the lines intersect, an error is raised.
            Default is False.

        """
        # Get list of lines from input
        if all(isinstance(e, Point) for e in elements):
            if len(elements) < 3:
                raise ValueError(
                    f"A {self.__class__.__name__} must have at least 3 points."
                )

            lines: list[Line] = lines_from_points(elements)  # type: ignore

        elif all(isinstance(e, Line) for e in elements):
            lines: list[Line] = list(elements)  # type: ignore

            if len(lines) < 2:
                raise ValueError(
                    f"A {self.__class__.__name__} must have at least 2 lines."
                )

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

        # Set points
        self._points = [ln[0] for ln in lines]
        if not self.is_closed():
            self._points.append(lines[-1][1])

        super().__init__(lines)

        # Raise error if intersecting but non-intersecting must be enforced
        # Do this after super init so that elements are set for
        # is_non_intersecting caching
        if strict_non_intersecting and not self.is_non_intersecting():
            raise ValueError(
                "The polyline is self intersecting. "
                "If this should not raise an error, "
                "set 'strict_non_intersecting' to False."
            )

    def get_self_intersections(self) -> tuple[list[Point], list[Line]]:
        """Return the intersections between the lines in the polyline.

        The points at which the lines meet (ie. mutual endpoints) DO NOT count as
        intersections.

        Returns
        -------
        tuple[list[Point], list[Line]]
            The first value is a list containing the points (excluding meeting points)
            that the lines have in common.
            The second value is a list containing the segments of that the lines have
            in common.

        """
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
        """Determine if the polyline is closed.

        Returns
        -------
        bool
            Whether the polyline is closed.

        """
        return self._is_closed

    @functools.cache
    def is_non_intersecting(self) -> bool:
        """Determine if the polyline is self-intersecting.

        Returns
        -------
        bool
            Whether the polyline is self-intersecting.

        """
        return not any(self.get_self_intersections())

    @property
    def points(self) -> list[Point]:
        """Get a list of points containing the points of the lines in the polyline.

        The points are ordered in the same manner as the lines.

        Returns
        -------
        list[Point]
            List of points of the lines in the polyline.

        """
        return self._points

    @override
    def __len__(self) -> int:
        return len(self._elements)

    def perimeter(self) -> float:
        """Calculate the perimeter of the polyline."""
        return round(sum(ln.length() for ln in self), CONFIG.precision)
