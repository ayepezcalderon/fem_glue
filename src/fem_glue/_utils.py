import math
from typing import Literal

from fem_glue._config import CONFIG

type _OP_STR = Literal["lt", "le", "eq", "ne", "ge", "gt"]


def tol_compare(a: float, b: float, op: _OP_STR, tol: float = CONFIG.tol) -> str:
    """Compare floats taking the given tolerance into account.

    Parameters
    ----------
    a : float
        The first float.
    b : float
        The second float.
    op : _OP_STR
        The comparison operator name. For example 'eq'.
    tol : float
        The tolerance to take into account.

    Returns
    -------
    bool
        The result of the comparison.

    """
    match op:
        case "lt":
            return a < b - tol
        case "le":
            return a < b + tol
        case "eq":
            return math.isclose(a, b, abs_tol=tol)
        case "ne":
            return not math.isclose(a, b, abs_tol=tol)
        case "ge":
            return a > b - tol
        case "gt":
            return a > b + tol
        case _:
            raise ValueError(f"Invalid comparison operator: {op}")
