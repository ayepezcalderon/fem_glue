from typing import Literal
from fem_glue._config import CONFIG

type _OP_STR = Literal["lt", "le", "eq", "ne", "ge", "gt"]


def tol_compare(a: float, b: float, op: _OP_STR, tol: float = CONFIG.tol) -> bool:
    match op:
        case "lt":
            return a < b - tol
        case "le":
            return a < b + tol
        case "eq":
            return abs(a - b) < tol
        case "ne":
            return not abs(a - b) < tol
        case "ge":
            return a > b - tol
        case "gt":
            return a > b + tol
        case _:
            raise ValueError(f"Invalid comparison operator: {op}")
