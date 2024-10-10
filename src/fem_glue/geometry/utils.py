import itertools
from collections.abc import Sequence

from fem_glue.geometry import Point
from fem_glue.geometry.dim1 import Line


def lines_from_points(points: Sequence[Point]) -> list[Line]:
    return [Line([p1, p2]) for p1, p2 in zip(points, points[1:], strict=False)]


def points_from_lines(lines: Sequence[Line]) -> list[Point]:
    return list(itertools.chain.from_iterable(lines))
