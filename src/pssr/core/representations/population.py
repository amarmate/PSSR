import logging
from typing import Optional, Any, Union, Sequence, Iterable

import numpy as np
import numpy.typing as npt

from pssr.core.representations.individual import Individual
from pssr.core.primitives import PrimitiveSet

Array = npt.NDArray[np.float64]

class Population:
    """
    Container for a population of GP individuals.
    
    The Population class manages a collection of Individual objects and provides
    vectorized operations for fitness evaluation and case-wise error calculations.
    
    Attributes
    ----------
    population : list[Individual]
        List of Individual objects in the population
    fitness : Optional[Array]
        Fitness values for each individual, shape (N,)
    errors_case : Optional[Array]
        Case-wise errors for each individual, shape (N, M) where N=individuals, M=test cases
    mad : Optional[Array]
        Median absolute deviation for each test case, shape (M,)
    combined_ranks : Optional[Array]
        Combined ranks for rank-based selection, shape (N,)
    """
    
    def __init__(self, population: list[Individual]):
            """
            Initialize a Population with a list of individuals.
            """
            if not population:
                raise ValueError("Population cannot be empty")
            
            self.population = population
            
            # Fitness arrays
            self.train_fitness: Optional[Array] = None
            self.test_fitness: Optional[Array] = None  
            
            # Semantic arrays (Outputs of the trees)
            self.train_semantics: Optional[Array] = None
            self.test_semantics: Optional[Array] = None
            
            # Error arrays (Strictly for training)
            self.errors_case: Optional[Array] = None
            
            # Primitive set (shared by all individuals)
            self.primitive_set : Optional[PrimitiveSet] = population[0].primitive_set
            
            # Slice indices for combined evaluation
            self.train_slice: Optional[slice] = None
            self.test_slice: Optional[slice] = None
        
    def __len__(self) -> int:
        return len(self.population)
    
    def __getitem__(self, index: int):
        return self.population[index]
    
    def __iter__(self):
        return iter(self.population)
    
    @property
    def size(self) -> int:
        return len(self.population)
    
    def set_slices(self, train_slice: slice, test_slice: Optional[slice] = None) -> None:
        """Set the slice indices for train/test portions of combined data."""
        for individual in self.population:
            individual.set_slices(train_slice, test_slice)
        self.train_slice = train_slice
        self.test_slice = test_slice
    
    def calculate_semantics(self, X: Array) -> None:
        """
        Calculate the semantics of the population on the full dataset (train + test combined).
        If slices are set via set_slices(), automatically assigns train_semantics
        and test_semantics from the combined result.
        
        Parameters
        ----------
        X : Array
            Combined input data (train + test)
            
        Returns
        -------
        None
        """
        if self.train_slice is None:
            logging.warning("Train slice not set. Calculating full semantics.")
            self.train_slice = slice(0, X.shape[0])
        
        all_semantics_list = [ind.calculate_semantics(X) for ind in self.population]

        full_semantics_matrix = np.stack(all_semantics_list)

        self.train_semantics = full_semantics_matrix[:, self.train_slice]

        if self.test_slice is not None:
            self.test_semantics = full_semantics_matrix[:, self.test_slice]
        else:
            self.test_semantics = None

        self.errors_case = None
        self.train_fitness = None
        self.test_fitness = None
            
    def calculate_errors_case(self, y: Array) -> None:
        """
        Calculate case-wise errors for all individuals (Absolute Errors).
        
        NOTE: This is strictly for TRAINING. It uses train_semantics and 
        stores the result in self.errors_case.
        
        Parameters
        ----------
        y : Array
            Combined target values (train + test). Will be sliced using train_slice.
        """
        if self.train_semantics is None:
            raise ValueError("Training semantics not calculated. Call calculate_semantics(X) first.")

        if self.errors_case is not None:
            logging.warning("Errors case already calculated.")
            return
        
        # Ensure train_slice is set
        if self.train_slice is None:
            # Default to full array if slice not set
            self.train_slice = slice(0, y.shape[0])
            
        # Vectorized calculation of Absolute Errors for Training
        # Shape: (N_individuals, N_samples)
        train_semantics = self.train_semantics.copy()
        train_slice = self.train_slice  # Type narrowing for linter
        self.errors_case = np.abs(train_semantics - y[train_slice])

    def standardize_errors(self, std_errs: bool = False) -> None: 
        """
        Standardize the errors_case matrix if requested.
        Only applies to Training data (self.errors_case).
        """
        if self.errors_case is None:
            raise ValueError("Errors case not calculated.")

        if std_errs:
            threshold = 1e-5
            mean = np.mean(self.errors_case, axis=0)
            stdev = np.std(self.errors_case, axis=0)
            
            standardized_errs = np.zeros_like(self.errors_case)
            
            mask = stdev > threshold
            standardized_errs[:, mask] = (self.errors_case[:, mask] - mean[mask]) / stdev[mask]
            
            self.errors_case = standardized_errs

    def evaluate(self, 
                 X: Optional[Array], 
                 y: Array) -> None:
        """
        Calculate fitness (RMSE) for all individuals for both training and testing data.
        """
        
        if self.errors_case is None and X is not None:
            logging.warning("[population.evaluate] Training semantics not calculated. Calculating now.")
            self.calculate_semantics(X)
            self.calculate_errors_case(y)
        
        if self.errors_case is None:
            raise ValueError("[population.evaluate] Errors case not calculated. Call calculate_errors_case(y) first.")
        
        # Calculate training fitness per individual: (n_individuals,)
        mse_train = np.mean(self.errors_case**2, axis=1)
        self.train_fitness = np.sqrt(mse_train)
        
        # Update individual fitness values
        assert self.train_fitness is not None  # Type narrowing for linter
        for i, individual in enumerate(self.population):
            individual.fitness = float(self.train_fitness[i])
        
        # Calculate test fitness if test data exists
        if self.test_semantics is not None and self.test_slice is not None:
            # test_semantics shape: (n_individuals, n_test_samples)
            # y[self.test_slice] shape: (n_test_samples,)
            # Broadcasting: (n_individuals, n_test_samples) - (n_test_samples,) -> (n_individuals, n_test_samples)
            test_slice = self.test_slice  # Type narrowing for linter
            test_errors = np.abs(self.test_semantics - y[test_slice])
            mse_test = np.mean(test_errors**2, axis=1)
            self.test_fitness = np.sqrt(mse_test)
            
            # Update individual test fitness values
            assert self.test_fitness is not None  # Type narrowing for linter
            for i, individual in enumerate(self.population):
                individual.test_fitness = float(self.test_fitness[i])
        else:
            self.test_fitness = None

    def get_best_individual(self) -> Any:
        """
        Get the individual with the best (lowest) training fitness.
        """
        if self.train_fitness is None:
            raise ValueError("Training fitness must be calculated before getting best individual.")
        
        best_idx = np.argmin(self.train_fitness)
        return self.population[best_idx]    
    
    def copy(self) -> "Population":
        """
        Create a deep copy of the population.
        """
        copied_individuals = [ind.copy() for ind in self.population]
        new_pop = Population(copied_individuals)
        
        # Copy arrays if they exist
        if self.train_fitness is not None:
            new_pop.train_fitness = self.train_fitness.copy()
        if self.test_fitness is not None:
            new_pop.test_fitness = self.test_fitness.copy()
        if self.errors_case is not None:
            new_pop.errors_case = self.errors_case.copy()
        if self.train_semantics is not None:
            new_pop.train_semantics = self.train_semantics.copy()
        if self.test_semantics is not None:
            new_pop.test_semantics = self.test_semantics.copy()
        
        return new_pop
    
    @classmethod
    def from_sequence(cls, pop: Union["Population", Sequence[Any]]) -> "Population":
        """
        Factory method to ensure we have a Population object.
        """
        if isinstance(pop, cls):
            return pop
        if isinstance(pop, Sequence):
            if not pop:
                raise ValueError("Population cannot be empty")
            # We assume objects in list are Individuals, but we don't strict type check 
            # to avoid circular imports if Individual is not imported here.
            return cls(list[Any](pop))
        raise TypeError(f"Unsupported population type: {type(pop)!r}")

    def fit_and_assess(self, X: Array, y: Array) -> dict[str, Any]:
        """
        Calculate jointly train and test fitness/errors and return a dictionary of generation metrics.
        """
        self.evaluate(X, y)
        
        if self.train_fitness is None:
            raise RuntimeError("Population fitness calculation failed")
        
        # Find best
        best_idx = int(np.argmin(self.train_fitness))
        best_ind = self.population[best_idx]
        
        # Helper to safely get attributes if they exist
        best_depth = getattr(best_ind, 'depth', 0)
        best_size = getattr(best_ind, 'total_nodes', 0)

        metrics = {
            # "fitness": self.train_fitness.copy(),
            # "errors_case": self.errors_case.copy() if self.errors_case is not None else None,
            "best_idx": best_idx,
            "best_fitness": float(self.train_fitness[best_idx]),
            "best_individual": best_ind,
            "mean_fitness": float(np.mean(self.train_fitness)),
            "std_fitness": float(np.std(self.train_fitness)),
            "worst_fitness": float(np.max(self.train_fitness)),
            "best_depth": float(best_depth),
            "best_size": float(best_size),
        }
        
        if self.test_fitness is not None:
            metrics["best_test_fitness"] = float(np.min(self.test_fitness))
            metrics["mean_test_fitness"] = float(np.mean(self.test_fitness))

        return metrics

    def extract_elites(self, n_elites: int) -> list:
        """
        Return a list of the top n_elites individuals (copies).
        """
        if n_elites <= 0:
            return []
        if self.train_fitness is None:
            raise ValueError("Fitness must be evaluated before extracting elites.")

        n_elites = min(n_elites, len(self.population))
        elite_indices = np.argsort(self.train_fitness)[:n_elites]
        
        return [self.population[idx].copy() for idx in elite_indices]

    def inject_elites(self, elites: Iterable[Any], rng: np.random.Generator) -> None:
        """
        Replace random individuals in the current population with the provided elites.
        """
        elites_list = list[Any](elites)
        if not elites_list:
            return

        n_replace = min(len(elites_list), len(self.population))
        
        # Randomly choose indices to overwrite
        replace_idx = rng.choice(len(self.population), size=n_replace, replace=False)

        for idx, elite in zip(replace_idx, elites_list, strict=True):
            self.population[idx] = elite.copy()
        
        # Invalidate fitness since population changed
        self.train_fitness = None
        self.errors_case = None
        self.train_semantics = None