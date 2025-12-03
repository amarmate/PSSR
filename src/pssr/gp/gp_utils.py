"""Utility functions for GP evolution, logging, and reporting."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import numpy.typing as npt

from pssr.core.representations.population import Population

Array = npt.NDArray[np.float64]
LogDict = Dict[str, List[float | int]]


def verbose_reporter(
    params: dict,
    first: bool = False,
    precision: int = 3,
    col_width: int = 14,
) -> None:
    """
    Prints a formatted report of custom parameters in a table format.
    
    Similar to the verbose_reporter in slim_gsgp_lib_np, this function prints
    a formatted table with headers (on first call) and values for each generation.
    
    Parameters
    ----------
    params : dict
        A dictionary containing key-value pairs of parameters to be reported.
    first : bool, default=False
        Whether this is the first report to be printed (prints headers).
    precision : int, default=3
        The number of decimal places to display for float values.
    col_width : int, default=14
        The width of the columns in the report.
    """
    if first:
        separator = ("+" + "-" * (col_width + 3)) * len(params) + "+"
        print(separator)
        print("".join([f"|{key.center(col_width+3)}" for key in params.keys()]) + "|")
        print(separator)

    # Print values
    values = []
    for value in params.values():
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif value is None:
            formatted_value = "None"
        else:
            formatted_value = str(value)
        values.append(formatted_value.center(col_width+3))
    
    print("|" + "".join([f"{value}|" for value in values]))
    print(('|' + '-' * (col_width + 3))*len(params) + '|')


def calculate_diversity(population: Population, X: Array) -> float:
    """
    Calculate population diversity using variance-based metric.
    
    Uses gsgp_pop_div_from_vectors_var: sqrt(2 * var(predictions))
    
    Parameters
    ----------
    population : Population
        Population to calculate diversity for
    X : Array
        Input data for evaluation
        
    Returns
    -------
    float
        Diversity metric value
    """
    semantics = population.train_semantics
    if semantics is None:
        raise ValueError("Train semantics not calculated. Call calculate_semantics() first.")
    return float(np.sqrt(2 * np.var(semantics)))

def init_log(include_test: bool = False) -> LogDict:
    """
    Initialize log dictionary for tracking evolution statistics.
    
    Parameters
    ----------
    include_test : bool, default=False
        Whether to include test fitness tracking in the log
    
    Returns
    -------
    LogDict
        Dictionary with empty lists for each metric
    """
    log = {
        "generation": [],
        "best_fitness": [],
        "mean_fitness": [],
        "worst_fitness": [],
        "std_fitness": [],
        "best_depth": [],
        "best_size": [],
        "eval_time": [],
    }
    if include_test:
        log["best_test_fitness"] = []
        log["mean_test_fitness"] = []
    return log


def update_log(log: LogDict, generation: int, metrics: dict, eval_time: float, test_metrics: dict | None = None) -> None:
    """
    Update log dictionary with generation metrics.
    
    Parameters
    ----------
    log : LogDict
        Log dictionary to update
    generation : int
        Generation number
    metrics : dict
        Metrics dictionary from population evaluation
    eval_time : float
        Evaluation time for this generation
    test_metrics : dict, optional
        Test metrics dictionary (if test evaluation was performed)
    """
    log["generation"].append(generation)
    log["best_fitness"].append(metrics["best_fitness"])
    log["mean_fitness"].append(metrics["mean_fitness"])
    log["worst_fitness"].append(metrics["worst_fitness"])
    log["std_fitness"].append(metrics["std_fitness"])
    log["best_depth"].append(metrics["best_depth"])
    log["best_size"].append(metrics["best_size"])
    log["eval_time"].append(float(eval_time))
    
    if "best_test_fitness" in metrics:
        log["best_test_fitness"].append(metrics["best_test_fitness"])
        log["mean_test_fitness"].append(metrics["mean_test_fitness"])


def print_verbose_report(
    generation: int,
    metrics: dict,
    eval_time: float,
    population: Population,
    X: Array,
    gen_time: float | None = None,
    mut_time: float | None = None,
    xo_time: float | None = None,
    selector_name: str | None = None,
    lex_rounds: float | None = None,
    first: bool = False,
) -> None:
    """
    Print verbose report for a generation.
    
    Matches the format from slim_gsgp_lib_np GP implementation.
    
    Parameters
    ----------
    generation : int
        Current generation number
    metrics : dict
        Metrics dictionary from population evaluation
    eval_time : float
        Time taken for evaluation
    population : Population
        Current population
    X : Array
        Input data for diversity calculation
    gen_time : float, optional
        Total time for generation (including selection, variation, etc.)
    mut_time : float, optional
        Time taken for mutation operations
    xo_time : float, optional
        Time taken for crossover operations
    selector_name : str, optional
        Name of the selector function (for lexicase rounds)
    lex_rounds : float, optional
        Average lexicase rounds if using lexicase selection
    first : bool
        Whether this is the first report (prints headers)
    """
    # Calculate diversity
    diversity = calculate_diversity(population, X)
    
    # Calculate average nodes
    avg_nodes = np.mean([ind.total_nodes for ind in population.population])
    
    # Build parameters dictionary matching reference format
    params = {
        "it": generation,
        "train": metrics["best_fitness"],
    }
    
    # Add test fitness right after train if available
    if "best_test_fitness" in metrics:
        params["test"] = metrics["best_test_fitness"]
    
    # Add remaining parameters
    params.update({
        "time": gen_time if gen_time is not None else eval_time,
        "nodes": int(metrics["best_size"]),
        "avg_nodes": int(avg_nodes),
        "div (var)": int(np.round(diversity)),
    })
    
    # Add mutation timing if available
    if mut_time is not None:
        params["mut"] = f"{np.round(1000 * mut_time, 2)}"
    else:
        params["mut"] = "N/A"
    
    # Add crossover timing if available
    if xo_time is not None:
        params["xo"] = f"{np.round(1000 * xo_time, 2)}"
    else:
        params["xo"] = "N/A"
    
    # Add lexicase rounds if using lexicase selection
    if selector_name and selector_name in ["els", "mels", "eplex"] and lex_rounds is not None:
        params["lex_r"] = np.round(lex_rounds, 3)
    
    # Print report
    verbose_reporter(params, first=first, precision=3, col_width=14)

