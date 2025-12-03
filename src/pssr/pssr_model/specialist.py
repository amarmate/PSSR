"""
Specialist class for PSSR (Piecewise Specialist Symbolic Regression).

A Specialist wraps a GP individual with cached semantics for fast 
ensemble evaluation. Specialists act as terminal nodes in ensemble trees.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from pssr.core.representations.individual import Individual

Array = npt.NDArray[np.float64]


class Specialist:
    """
    Wrapper for a GP individual that caches its semantics.
    
    In PSSR, specialists are pre-trained GP individuals whose outputs
    are cached for fast lookup during ensemble evolution. They act as
    terminal nodes in the ensemble tree structure.
    
    Semantics are stored as a single combined array (train + test) with
    a cutoff point, similar to how GPRegressor handles it for efficiency.
    
    Attributes
    ----------
    name : str
        Unique identifier for this specialist (e.g., "S_0", "S_1")
    individual : Individual
        The underlying GP individual
    semantics : Optional[Array]
        Cached predictions on combined data (train + test), shape (n_total_samples,)
    fitness : Optional[float]
        Training fitness (RMSE) of the underlying individual
    test_fitness : Optional[float]
        Test fitness (RMSE) of the underlying individual
    nodes_count : int
        Number of nodes in the underlying GP tree
    depth : int
        Depth of the underlying GP tree
    """
    
    def __init__(
        self,
        name: str,
        individual: Individual,
        semantics: Optional[Array] = None,
    ):
        """
        Initialize a Specialist.
        
        Parameters
        ----------
        name : str
            Unique identifier for this specialist
        individual : Individual
            The GP individual to wrap
        semantics : Optional[Array]
            Pre-computed combined predictions (train + test).
        """
        self.name = name
        self.individual = individual
        
        # Combined semantics (train + test in one array)
        self.semantics = semantics
        
        # Copy fitness metrics from individual
        self.fitness = individual.fitness
        self.test_fitness = individual.test_fitness
        
        # Copy tree metrics
        self.nodes_count = individual.total_nodes
        self.depth = individual.depth
    
    def compute_semantics(self, X: Array) -> None:
        """
        Compute and cache combined semantics.
        
        Parameters
        ----------
        X : Array
            Combined input data (train + test), shape (n_total_samples, n_features)
        """
        self.semantics = self.individual.predict(X)
    
    def predict(self, X: Array) -> Array:
        """
        Make predictions on new data.
        
        This always computes fresh predictions, not using cached semantics.
        
        Parameters
        ----------
        X : Array
            Input data, shape (n_samples, n_features)
            
        Returns
        -------
        Array
            Predictions, shape (n_samples,)
        """
        return self.individual.predict(X)
    
    def __repr__(self) -> str:
        fitness_str = f", fitness={self.fitness:.6f}" if self.fitness is not None else ""
        return f"Specialist({self.name}, nodes={self.nodes_count}{fitness_str})"
    
    def __str__(self) -> str:
        return self.name


def create_specialists_from_population(
    population,
    X: Array,
    prefix: str = "S_",
) -> dict[str, Specialist]:
    """
    Create a dictionary of specialists from a GP population.
    
    This function takes a population of GP individuals and wraps each
    as a Specialist with pre-computed combined semantics.
    
    Parameters
    ----------
    population : Population or list[Individual]
        The GP population to convert to specialists
    X : Array
        Combined data (train + test) for computing semantics
    prefix : str
        Prefix for specialist names (default "S_")
        
    Returns
    -------
    dict[str, Specialist]
        Dictionary mapping specialist names to Specialist objects
    """
    specialists = {}
    
    # Handle both Population objects and lists
    if hasattr(population, 'population'):
        individuals = population.population
    else:
        individuals = list(population)
    
    for i, individual in enumerate(individuals):
        name = f"{prefix}{i}"
        specialist = Specialist(name=name, individual=individual)
        
        # Compute combined semantics
        specialist.compute_semantics(X)
            
        specialists[name] = specialist
    
    return specialists
