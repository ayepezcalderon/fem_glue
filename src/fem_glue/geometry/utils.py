"""Utility convinience functions for geometries.

A collection of common operations and algorithms that do not belong to any single
type of geometry.
"""

from collections.abc import Sequence

from fem_glue.geometry import Point
from fem_glue.geometry.dim1 import Line


def lines_from_points(points: Sequence[Point]) -> list[Line]:
    """Get a list of lines from a sequence of points.

    Point n defines the end point of line n-1 and the start point of line n.
    The exception to this rule is when the point is the first or last point of the
    sequence. For the first point the start point only defines the the start point of
    the first line, and for the last point the end point only defines the end point of
    the last line. These first/last

    Parameters
    ----------
    points : Sequence[Point]
        The input sequence of points.

    Returns
    -------
    list[Line]
        The output sequence of lines obtained from the input sequence of points.

    """
    return [Line([p1, p2]) for p1, p2 in zip(points, points[1:], strict=False)]
