"""
Vectorized fitness functions for GP evolution.

All fitness functions are designed to work with population-level semantics:
- Input: y_true (n_samples,) and y_pred (n_individuals, n_samples)
- Output: fitness values (n_individuals,)

For minimization, lower values are better. R² is converted to 1-R² for this purpose.
"""

import logging
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]


def rmse(y_true: Array, y_pred: Array) -> Array:
    """
    Root Mean Squared Error (vectorized for population).
    
    Parameters
    ----------
    y_true : Array
        True target values, shape (n_samples,)
    y_pred : Array
        Predicted values, shape (n_individuals, n_samples)
        
    Returns
    -------
    Array
        RMSE for each individual, shape (n_individuals,)
    """
    # y_pred shape: (n_individuals, n_samples)
    # y_true shape: (n_samples,)
    # Broadcasting: (n_individuals, n_samples) - (n_samples,) -> (n_individuals, n_samples)
    errors = y_pred - y_true
    mse_values = np.mean(errors**2, axis=1)
    return np.sqrt(mse_values)


def mse(y_true: Array, y_pred: Array) -> Array:
    """
    Mean Squared Error (vectorized for population).
    
    Parameters
    ----------
    y_true : Array
        True target values, shape (n_samples,)
    y_pred : Array
        Predicted values, shape (n_individuals, n_samples)
        
    Returns
    -------
    Array
        MSE for each individual, shape (n_individuals,)
    """
    errors = y_pred - y_true
    return np.mean(errors**2, axis=1)


def mae(y_true: Array, y_pred: Array) -> Array:
    """
    Mean Absolute Error (vectorized for population).
    
    Parameters
    ----------
    y_true : Array
        True target values, shape (n_samples,)
    y_pred : Array
        Predicted values, shape (n_individuals, n_samples)
        
    Returns
    -------
    Array
        MAE for each individual, shape (n_individuals,)
    """
    errors = np.abs(y_pred - y_true)
    return np.mean(errors, axis=1)


def r2(y_true: Array, y_pred: Array) -> Array:
    """
    R² (Coefficient of Determination) for maximization.
    
    Returns actual R², where higher values are better (1.0 = perfect fit).
    Note: This metric works with MAXIMIZATION, not minimization like others.
    
    Parameters
    ----------
    y_true : Array
        True target values, shape (n_samples,)
    y_pred : Array
        Predicted values, shape (n_individuals, n_samples)
        
    Returns
    -------
    Array
        R² for each individual, shape (n_individuals,)
        Values range from -infinity (poor) to 1.0 (perfect).
    """
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    # Residual sum of squares for each individual
    # (n_individuals, n_samples) - (n_samples,) -> (n_individuals, n_samples)
    residuals = y_pred - y_true
    ss_res = np.sum(residuals**2, axis=1)
    
    # R² = 1 - (SS_res / SS_tot)
    if ss_tot == 0:
        # Handle edge case: constant y_true
        return np.ones(y_pred.shape[0])
    
    return 1.0 - (ss_res / ss_tot)


# Registry of fitness functions
# Each entry is (function, maximize) where maximize=True means higher is better
FITNESS_REGISTRY: dict[str, tuple[Callable[[Array, Array], Array], bool]] = {
    "rmse": (rmse, False),  # minimize
    "mse": (mse, False),    # minimize
    "mae": (mae, False),    # minimize
    "r2": (r2, True),       # maximize (higher R² is better)
}


def fetch_fitness(name: str) -> Callable[[Array, Array], Array]:
    """
    Get a fitness function by name.
    
    Parameters
    ----------
    name : str
        Name of the fitness function. Options: "rmse", "mse", "mae", "r2"
        
    Returns
    -------
    Callable
        Fitness function that accepts (y_true, y_pred) and returns fitness values.
        
    Raises
    ------
    ValueError
        If the fitness function name is not recognized.
        
    Notes
    -----
    Use `is_maximization(name)` to check if higher values are better.
    """
    if name not in FITNESS_REGISTRY:
        available = ", ".join(sorted(FITNESS_REGISTRY.keys()))
        raise ValueError(f"Unknown fitness function '{name}'. Available: {available}")
    
    return FITNESS_REGISTRY[name][0]


def is_maximization(name: str) -> bool:
    """
    Check if a fitness function uses maximization (higher is better).
    
    Parameters
    ----------
    name : str
        Name of the fitness function.
        
    Returns
    -------
    bool
        True if higher values are better (maximization), False if lower is better.
        
    Raises
    ------
    ValueError
        If the fitness function name is not recognized.
    """
    if name not in FITNESS_REGISTRY:
        available = ", ".join(sorted(FITNESS_REGISTRY.keys()))
        raise ValueError(f"Unknown fitness function '{name}'. Available: {available}")
    
    return FITNESS_REGISTRY[name][1]


def list_fitness_functions() -> list[str]:
    """
    List all available fitness function names.
    
    Returns
    -------
    list[str]
        List of available fitness function names.
    """
    return list(FITNESS_REGISTRY.keys())


def register_fitness(
    name: str,
    func: Callable[[Array, Array], Array],
    maximize: bool = False,
) -> None:
    """
    Register a custom fitness function.
    
    The function must accept (y_true, y_pred) where:
    - y_true has shape (n_samples,)
    - y_pred has shape (n_individuals, n_samples)
    
    And return fitness values with shape (n_individuals,).
    
    Parameters
    ----------
    name : str
        Name to register the function under.
    func : Callable
        Fitness function to register.
    maximize : bool, default=False
        If True, higher values are better (maximization).
        If False, lower values are better (minimization).
    """
    FITNESS_REGISTRY[name] = (func, maximize)
    direction = "maximization" if maximize else "minimization"
    logger.info(f"Registered custom fitness function: {name} ({direction})")

