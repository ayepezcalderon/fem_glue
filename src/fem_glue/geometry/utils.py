from collections.abc import Sequence
from fem_glue.geometry import Line, Point


def lines_from_points(points: Sequence[Point]) -> list[Line]:
    return [Line([p1, p2]) for p1, p2 in zip(points, points[1:])]


def points_from_polyline_lines(
    lines: Sequence[Line], closed: bool = False
) -> list[Point]:
    if not all(l1[1] == l2[0] for l1, l2 in zip(lines, lines[1:])):
        raise ValueError("The lines are not connected.")

    points = [line[0] for line in lines]
    if not closed:
        points.append(lines[-1][1])

    return points
