import math
from typing import Any, Literal, TypeAliasType, get_args

from fem_glue._config import CONFIG

type _OP_STR = Literal["lt", "le", "eq", "ne", "ge", "gt"]


def tol_compare(a: float, b: float, op: _OP_STR, tol: float = CONFIG.tol) -> bool:
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


def check_literal(name: str, value: Any, literal_type_alias: TypeAliasType) -> None:
    literal_args = get_args(literal_type_alias.__value__)
    if value not in literal_args:
        possibles_clause = ", ".join(literal_args[:-1]) + " or " + literal_args[-1]
        raise ValueError(f"The parameter {name} must be one of {possibles_clause}.")
