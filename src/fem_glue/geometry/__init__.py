"""A geometry package for shapes in 3D space.

Defines points, 1D and 2D shapes in 3D space.
Defines relations and operations between these geometrical entities.
"""

from fem_glue.geometry.dim0 import Point
from fem_glue.geometry.dim1 import Line, Polyline
from fem_glue.geometry.dim2 import Polygon

__all__ = ["Point", "Line", "Polyline", "Polygon"]
