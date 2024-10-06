class PointOnShapeError(Exception):
    """Raise when a point is on a shape and this is not allowed."""


class PointNotOnShapeError(Exception):
    """Raise when a point was expected to be on a shape but is not."""
