import numpy as np

from typing import overload
from collections.abc import Sequence

from fem_glue.geometry.dim1.polyline import Polyline, Point, Line
from fem_glue.geometry._bases import SequentialGeometry
from fem_glue._config import CONFIG


class Surface(SequentialGeometry[Line]):
    """
    A planar surface in 3D space.
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
        # Surface has closed and non-self-intersecting polyline as boundary
        self.boundary = Polyline(
            boundary_elements, close=True, strict_non_intersecting=True
        )

        # Surface has same sequential geometry behavior as its boundary
        super().__init__(self.boundary)

        # Set orthogonal basis of the plane
        self.basis, _basis_computation_lines = self._find_orthogonal_basis()
        self.tangents = self.basis[:, :2]
        self.normal = self.basis[:, 2]

        # Check if all the lines in the boundary are coplanar
        for line in self:
            if line in _basis_computation_lines:
                continue

            if not self.is_tangent_to_line(line):
                raise ValueError(
                    f"{line} is not tangent to {_basis_computation_lines}."
                )

    def is_tangent_to_line(self, line: Line) -> bool:
        """
        Check if the surface is tangent to the given line.

        Parameters
        ----------
        line : Line
            The line to check against.

        Returns
        -------
        bool
            True if the surface is tangent to the line, False otherwise.
        """
        return bool(np.dot(self.normal, line.dir_vector() < CONFIG.tol))

    def _find_orthogonal_basis(self) -> tuple[np.ndarray, tuple[Line, Line]]:
        """
        Find an orthogonal unit basis for the plane of the surface and the Lines
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
        # Find the first two non-parallel lines in the surface's boundary
        first_computation_line = self[0]
        try:
            second_computation_line = next(
                line
                for line in self[1:]
                if not line.is_parallel(first_computation_line)
            )
        except StopIteration:
            raise ValueError("All lines in the surface's boundary are parallel.")

        # Find unit vector tangent to plane and parallel to first computation line
        unit_tangent_1 = first_computation_line.dir_unit_vector()

        # Find normal unit vector to plane
        normal = np.cross(
            unit_tangent_1,
            second_computation_line.dir_vector(),
        )
        unit_normal = normal / np.linalg.norm(normal)

        # Find unit vector tangent to plane and orthogonal to first computation line
        tangent_2 = np.cross(unit_normal, first_computation_line.dir_vector())
        unit_tangent_2 = tangent_2 / np.linalg.norm(tangent_2)

        # Return orthogonal basis
        return np.array([unit_tangent_1, unit_tangent_2, unit_normal]).T, (
            first_computation_line,
            second_computation_line,
        )
