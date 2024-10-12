import functools
import math
from collections.abc import Sequence
from typing import overload

import numpy as np

from fem_glue._config import CONFIG
from fem_glue.geometry._bases import SequentialGeometry
from fem_glue.geometry.dim1.polyline import Line, Point, Polyline


class Polygon(SequentialGeometry[Line]):
    """A planar polygon in 3D space.
    The boundary must be a closed and non-self-intersecting polyline.
    Each line in the boundary must lie in the same 2D plane in 3D space.

    Attributes
    ----------
    boundary : Polyline
        The boundary of the polygon.
    tangents : np.ndarray
        2x3 array. Contains two 1x3 sub-arrays that define unit vectors tangent to the
        plane of the polygon and are orthogonal to each other.
    normal : np.ndarray
        1x3 array that defines the normal unit vector to the plane of the polygon.
    basis : np.ndarray
        3x3 array. The first two sub-arrays correspond to "tangents", and the
        third corresponds to "normal". The 3 together define the orthogonal basis
        for the local coordinate system of the plane of the polygon.

    """

    @overload
    def __init__(
        self,
        boundary_elements: Sequence[Line],
        /,
    ): ...

    @overload
    def __init__(
        self,
        boundary_elements: Sequence[Point],
        /,
    ): ...

    def __init__(
        self,
        boundary_elements: Sequence[Line] | Sequence[Point],
        /,
    ):
        # Polygon has closed and non-self-intersecting polyline as boundary
        self.boundary = Polyline(
            boundary_elements, close=True, strict_non_intersecting=True
        )

        # Polygon has same sequential geometry behavior as its boundary
        super().__init__(self.boundary)

        # Set orthogonal basis of the plane
        self.basis, _basis_computation_lines = self._find_orthogonal_basis()
        self.tangents = self.basis[:, :2]
        self.normal = self.basis[:, 2]

        # Check if all the lines in the boundary are coplanar
        for line in self:
            if line in _basis_computation_lines:
                continue

            if not self.line_is_tangent(line):
                raise ValueError(
                    f"{line} is not tangent to {_basis_computation_lines}."
                )

    @functools.cache
    def get_plane_coefficients(self) -> tuple[float, float, float, float]:
        """Get the coefficients of the plane equation a, b, c, d.
        The equation is defined as a*x + b*y + c*z + d = 0.

        Returns
        -------
        tuple[float, float, float, float]
            The coefficients of the plane equation.

        """
        a, b, c = self.normal
        d = -np.dot(self.normal, np.array(Point(self.boundary.points[0])))
        return a, b, c, d

    def point_on_polygon_plane(self, point: Point) -> bool:
        """Check if the point is in the same plane as the polygon.

        Parameters
        ----------
        point : Point
            The point to check.

        Returns
        -------
        bool
            True if the point is in the same plane as the polygon, False otherwise.

        """
        # If the line formed by the given point and a vertex of the polygon is parallel
        # to the polygon, then the given point is on the same plane as the polygon
        # Edge case: if the point is equal to the sampled vertex, point is on plane
        vertex = self.boundary.points[0]
        return point == vertex or self.line_is_tangent(Line([point, vertex]))

    def point_on_polygon_boundary(self, point: Point) -> bool:
        """Check if the point is on the boundary of the polygon.

        Parameters
        ----------
        point : Point
            The point to check.

        Returns
        -------
        bool
            True if the point is on the boundary of the polygon, False otherwise.

        """
        # If the point is not on the plane of the polygon, it is not on the boundary
        if not self.point_on_polygon_plane(point):
            return False

        # Check if the point is on the boundary of the polygon
        for edge in self.boundary:
            if point in edge:
                return True

        return False

    def point_inside_polygon(self, point: Point) -> bool:
        """Check if the point is inside the polygon (ie. inside its boundary).
        NOTE: If the point is on the boundary of the polygon, it is not inside of it.

        Parameters
        ----------
        point : Point
            The point to check.

        Returns
        -------
        bool
            True if the point is inside the polygon, False otherwise.

        """
        # If the point is not on the plane of the polygon, it is not inside the polygon
        if not self.point_on_polygon_plane(point):
            return False

        # Check if the point is inside the polygon
        # Ray casting algorithm
        # https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm
        # Cast a ray from the point in the positive x direction
        # Count the number of intersections with the polygon boundary
        # If the number of intersections is odd, the point is inside the polygon
        # If the number of intersections is even, the point is outside the polygon
        ray = Line([point, Point([point[0] + 1, point[1], point[2]])])
        intersections = 0
        for edge in self.boundary:
            if ray.intersect(edge):
                intersections += 1

        return intersections % 2 == 1

    def line_is_tangent(self, line: Line) -> bool:
        """Check if the polygon is tangent to the given line.

        Parameters
        ----------
        line : Line
            The line to check against.

        Returns
        -------
        bool
            True if the polygon is tangent to the line, False otherwise.

        """
        return math.isclose(np.dot(self.normal, line.as_vector()), CONFIG.tol)

    def _find_orthogonal_basis(self) -> tuple[np.ndarray, tuple[Line, Line]]:
        """Find an orthogonal unit basis for the plane of the polygon and the Lines
        used to compute this basis.

        The first return value is the basis matrix.
        The first two columns of the basis are tangent to the plane, the third is
        normal.

        The second return value are the lines used to compute the basis.

        Returns
        -------
        tuple[np.ndarray, tuple[Line, Line]]
            The orthogonal basis and the lines used to compute it.

        """
        # Find the first two non-parallel lines in the polygon's boundary
        first_computation_line = self[0]
        try:
            second_computation_line = next(
                line
                for line in self[1:]
                if not line.is_parallel(first_computation_line)
            )
        except StopIteration as e:
            raise ValueError("All lines in the polygon's boundary are parallel.") from e

        # Find unit vector tangent to plane and parallel to first computation line
        unit_tangent_1 = first_computation_line.dir_unit_vector()

        # Find normal unit vector to plane
        normal = np.cross(
            unit_tangent_1,
            second_computation_line.as_vector(),
        )
        unit_normal = normal / np.linalg.norm(normal)

        # Find unit vector tangent to plane and orthogonal to first computation line
        tangent_2 = np.cross(unit_normal, first_computation_line.as_vector())
        unit_tangent_2 = tangent_2 / np.linalg.norm(tangent_2)

        # Return orthogonal basis
        return np.array([unit_tangent_1, unit_tangent_2, unit_normal]).T, (
            first_computation_line,
            second_computation_line,
        )
