class PointOnLineError(Exception):
    """Raise when a point is on a line and this is not allowed."""


class PointNotOnLineError(Exception):
    """Raise when a point was expected to be on a line but is not."""
