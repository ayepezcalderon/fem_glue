from typing import override, Self
from collections.abc import Iterable

from fem_glue.geometry.geometry import Geometry
from fem_glue.geometry import Line


class Polyline(Geometry[Line]):
    """
    A closed non-intersecting polyline with n-straight lines.
    """
    def __init__(self, elements: Iterable[Line], /):
        super().__init__(elements)

        # Validate that the polyline is closed and non-intersecting.
        if self[0] != self[-1]:
            raise ValueError("The polyline is not closed.")

        for i, ln1 in enumerate(self):
            for ln2 in self[i + 1:]:
                if ln1.intersects(ln2):
                    raise ValueError("The polyline is intersecting.")


    @override
    def __len__(self) -> int:
        return len(self._elements)

    def perimeter(self) -> float:
        """
        Calculate the perimeter of the polyline.
        """
        return sum(ln.length() for ln in self)

    def area(self) -> float: ...

    def normalize(self) -> Self:
        """
        Normalize the polyline such that it has unit area.
        """

