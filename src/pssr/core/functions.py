from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

@dataclass(frozen=True)
class GPFunction:
    """
    Represents a Genetic Programming function with a name, callable, and arity.
    """
    name: str
    func: Callable[..., Array]
    arity: int
    
def _safe_div(x: Array, y: Array) -> Array:
    return np.where(
        np.abs(y) > 0.00001, 
        np.divide(x, y, out=np.zeros_like(x), where=np.abs(y) > 0.00001),
        1.0
    )

def _safe_sqrt(x: Array) -> Array:
    return np.sqrt(np.abs(x))

def _safe_log(x: Array) -> Array:
    return np.log(np.abs(x) + 1e-10)

def _aq_numpy(x: Array, y: Array) -> Array:
    return np.divide(x, np.sqrt(1 + y**2))


DEFAULT_FUNCTIONS: dict[str, GPFunction] = {
    "add": GPFunction("add", np.add, arity=2),
    "sub": GPFunction("sub", np.subtract, arity=2),
    "mul": GPFunction("mul", np.multiply, arity=2),
    "div": GPFunction("div", _safe_div, arity=2),
    "truediv": GPFunction("truediv", np.divide, arity=2),
    "sqrt": GPFunction("sqrt", _safe_sqrt, arity=1),
    "aq": GPFunction("aq", _aq_numpy, arity=2),
    "neg": GPFunction("neg", np.negative, arity=1),
    "sin": GPFunction("sin", np.sin, arity=1),
    "cos": GPFunction("cos", np.cos, arity=1),
    "exp": GPFunction("exp", np.exp, arity=1),
    "log": GPFunction("log", _safe_log, arity=1),
    "pow": GPFunction("pow", np.power, arity=2),
    "max": GPFunction("max", np.maximum, arity=2),
    "min": GPFunction("min", np.minimum, arity=2),
}

