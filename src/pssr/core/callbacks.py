"""
Callback interface for GP evolution.

Callbacks allow monitoring and interfering with the optimization process.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pssr.core.representations.individual import Individual
from pssr.core.representations.population import Population

Array = npt.NDArray[np.float64]


class Callback(ABC):
    """
    Abstract base class for callbacks.
    
    Callbacks are called at specific points during evolution:
    - on_generation_end: Called after each generation completes
    - on_evolution_end: Called when evolution finishes
    
    Callbacks can stop evolution early by returning True from on_generation_end.
    """
    
    @abstractmethod
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        metrics: dict[str, Any],
    ) -> bool:
        """
        Called at the end of each generation.
        
        Parameters
        ----------
        generation : int
            Current generation number (0-indexed)
        population : Population
            Current population after this generation
        best_individual : Individual
            Best individual found so far (across all generations)
        best_fitness : float
            Fitness of the best individual
        metrics : dict
            Dictionary of metrics from this generation including:
            - "mean_fitness": Mean fitness of population
            - "best_fitness": Best fitness this generation
            - "worst_fitness": Worst fitness this generation
            - "std_fitness": Standard deviation of fitness
            
        Returns
        -------
        bool
            True to stop evolution early, False to continue
        """
        pass
    
    def on_evolution_end(
        self,
        generations_completed: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        early_stopped: bool,
    ) -> None:
        """
        Called when evolution ends.
        
        Parameters
        ----------
        generations_completed : int
            Total number of generations completed
        population : Population
            Final population
        best_individual : Individual
            Best individual found
        best_fitness : float
            Fitness of the best individual
        early_stopped : bool
            Whether evolution was stopped early (by callback)
        """
        pass


class EarlyStoppingCallback(Callback):
    """
    Stop evolution when fitness stops improving.
    
    Parameters
    ----------
    patience : int
        Number of generations without improvement before stopping.
    min_delta : float
        Minimum change in fitness to qualify as an improvement.
    mode : str
        'min' for minimization problems, 'max' for maximization.
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best_fitness: Optional[float] = None
        self.stopped_generation: Optional[int] = None
    
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        metrics: dict[str, Any],
    ) -> bool:
        current_fitness = metrics["best_fitness"]
        
        if self.best_fitness is None:
            self.best_fitness = current_fitness
            return False
        
        if self.mode == "min":
            improved = current_fitness < self.best_fitness - self.min_delta
        else:  # max
            improved = current_fitness > self.best_fitness + self.min_delta
        
        if improved:
            self.best_fitness = current_fitness
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_generation = generation
            return True
        
        return False
    
    def on_evolution_end(
        self,
        generations_completed: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        early_stopped: bool,
    ) -> None:
        if self.stopped_generation is not None:
            pass  # Could log: f"Early stopping at generation {self.stopped_generation}"


class LoggingCallback(Callback):
    """
    Log evolution progress to a list.
    
    Access the log via the `history` attribute after evolution.
    """
    
    def __init__(self):
        self.history: list[dict[str, Any]] = []
    
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        metrics: dict[str, Any],
    ) -> bool:
        entry = {
            "generation": generation,
            "best_fitness": best_fitness,
            "gen_best_fitness": metrics["best_fitness"],
            "mean_fitness": metrics["mean_fitness"],
            "worst_fitness": metrics["worst_fitness"],
            "std_fitness": metrics["std_fitness"],
        }
        self.history.append(entry)
        return False
    
    def on_evolution_end(
        self,
        generations_completed: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        early_stopped: bool,
    ) -> None:
        self.history.append({
            "event": "evolution_end",
            "generations_completed": generations_completed,
            "final_best_fitness": best_fitness,
            "early_stopped": early_stopped,
        })


class FitnessThresholdCallback(Callback):
    """
    Stop evolution when fitness reaches a target threshold.
    
    Parameters
    ----------
    threshold : float
        Target fitness value.
    mode : str
        'min' stops when fitness <= threshold, 'max' when fitness >= threshold.
    """
    
    def __init__(self, threshold: float, mode: str = "min"):
        self.threshold = threshold
        self.mode = mode
        self.reached_generation: Optional[int] = None
    
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        metrics: dict[str, Any],
    ) -> bool:
        if self.mode == "min":
            reached = best_fitness <= self.threshold
        else:
            reached = best_fitness >= self.threshold
        
        if reached:
            self.reached_generation = generation
            return True
        
        return False


class CompositeCallback(Callback):
    """
    Combine multiple callbacks.
    
    Evolution stops if any callback returns True.
    """
    
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks
    
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        metrics: dict[str, Any],
    ) -> bool:
        for callback in self.callbacks:
            if callback.on_generation_end(
                generation, population, best_individual, best_fitness, metrics
            ):
                return True
        return False
    
    def on_evolution_end(
        self,
        generations_completed: int,
        population: Population,
        best_individual: Individual,
        best_fitness: float,
        early_stopped: bool,
    ) -> None:
        for callback in self.callbacks:
            callback.on_evolution_end(
                generations_completed, population, best_individual, best_fitness, early_stopped
            )

