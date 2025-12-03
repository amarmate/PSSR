"""
PSSRegressor: Piecewise Specialist Symbolic Regression.

This module provides the main PSSRegressor class which implements
the Multi-SLIM approach for symbolic regression using piecewise
specialists with conditional routing.
"""

from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import RegressorMixin

from pssr.core.normalization import NormalizationMixin
from pssr.core.primitives import FunctionSet, PrimitiveSet
from pssr.core.representations.population import Population
from pssr.core.selection import fetch_selector
from pssr.gp.gp_regressor import GPRegressor
from pssr.pssr_model.specialist import Specialist, create_specialists_from_population
from pssr.pssr_model.ensemble_individual import EnsembleIndividual
from pssr.pssr_model.ensemble_initialization import (
    ensemble_initializer,
    fetch_ensemble_initializer,
)
from pssr.pssr_model.ensemble_operators import (
    fetch_ensemble_crossover,
    fetch_ensemble_mutation,
    ensemble_variator,
)


def _calculate_ensemble_semantics_combined(
    population: Population,
    train_slice: slice,
    test_slice: Optional[slice] = None,
) -> None:
    """
    Calculate combined semantics for all ensemble individuals and slice into train/test.
    
    This function mirrors Population.calculate_semantics_combined but for EnsembleIndividuals
    which use cached primitive_set values instead of X.
    
    Sets slices on each individual so they auto-assign train/test semantics.
    """
    train_semantics_list = []
    test_semantics_list = []
    
    for ind in population.population:
        # Reset cached semantics
        ind.train_semantics = None
        ind.test_semantics = None
        
        # Set slices on individual so calculate_semantics auto-assigns train/test
        ind.set_slices(train_slice, test_slice)
        
        # This now auto-assigns train_semantics and test_semantics
        ind.calculate_semantics(None)
        
        train_semantics_list.append(ind.train_semantics)
        
        if test_slice is not None and ind.test_semantics is not None:
            test_semantics_list.append(ind.test_semantics)
    
    # Update population-level semantics arrays
    population.train_semantics = np.stack(train_semantics_list)
    if test_slice is not None and test_semantics_list:
        population.test_semantics = np.stack(test_semantics_list)
    else:
        population.test_semantics = None
    
    # Reset cached fitness/errors
    population.train_fitness = None
    population.test_fitness = None
    population.errors_case = None


class PSSRegressor(NormalizationMixin, RegressorMixin):
    """
    Piecewise Specialist Symbolic Regression (PSSR) Regressor.
    
    PSSR is a two-phase approach:
    1. Train a population of GP specialists
    2. Evolve conditional routing trees that combine specialists
    
    The resulting model partitions the input space using learned conditions
    and routes inputs to appropriate specialist models.
    
    Parameters
    ----------
    # Specialist training parameters
    specialist_pop_size : int
        Population size for specialist GP evolution
    specialist_n_gen : int
        Number of generations for specialist training
    specialist_max_depth : int
        Maximum tree depth for specialists
    specialist_init_depth : int
        Initial tree depth for specialists
    
    # Ensemble evolution parameters
    ensemble_pop_size : int
        Population size for ensemble evolution
    ensemble_n_gen : int
        Number of generations for ensemble evolution
    ensemble_max_depth : int
        Maximum depth for ensemble trees
    depth_condition : int
        Maximum depth for condition trees in ensemble
    
    # Variation parameters
    p_xo : float
        Probability of crossover vs mutation
    
    # General parameters
    random_state : int
        Random seed for reproducibility
    normalize : bool
        Whether to normalize X and y
    
    # Function set
    functions : Optional[FunctionSet]
        Functions for GP trees (default: add, sub, mul, div)
    condition_functions : Optional[FunctionSet]
        Functions for condition trees (default: same as functions)
    constant_range : float
        Range for constant terminals
    
    # Selection
    selector : Union[str, Callable]
        Selection method for both phases
    
    Attributes
    ----------
    specialists_ : dict[str, Specialist]
        Trained specialists after fitting
    best_ensemble_ : EnsembleIndividual
        Best ensemble individual after fitting
    specialist_population_ : Population
        Final specialist population
    ensemble_population_ : Population
        Final ensemble population
    log_ : dict
        Training log with fitness history
    """
    
    def __init__(
        self,
        # Specialist parameters
        specialist_pop_size: int = 100,
        specialist_max_depth: int = 6,
        specialist_init_depth: int = 2,
        
        # Ensemble parameters
        ensemble_pop_size: int = 100,
        ensemble_max_depth: int = 4,
        depth_condition: int = 3,
        
        # Variation parameters
        p_xo: float = 0.5,
        
        # General parameters
        random_state: int = 42,
        normalize: bool = False,
        
        # Function sets
        functions: Optional[Union[list[str], FunctionSet]] = None,
        condition_functions: Optional[Union[list[str], FunctionSet]] = None,
        constant_range: float = 1.0,
        
        # Selection
        selector: Union[str, Callable] = "tournament",
        
        # Scalers
        X_scaler=None,
        y_scaler=None,
        
        **params,
    ) -> None:
        
        NormalizationMixin.__init__(
            self, normalize=normalize, X_scaler=X_scaler, y_scaler=y_scaler
        )
        
        # Validate parameters
        if specialist_pop_size < 1:
            raise ValueError(f"specialist_pop_size must be >= 1, got {specialist_pop_size}")
        if ensemble_pop_size < 1:
            raise ValueError(f"ensemble_pop_size must be >= 1, got {ensemble_pop_size}")
        
        # Specialist parameters
        self.specialist_pop_size = specialist_pop_size
        self.specialist_max_depth = specialist_max_depth
        self.specialist_init_depth = specialist_init_depth
        
        # Ensemble parameters
        self.ensemble_pop_size = ensemble_pop_size
        self.ensemble_max_depth = ensemble_max_depth
        self.depth_condition = depth_condition
        
        # Variation
        self.p_xo = p_xo
        
        # General
        self.random_state = random_state
        self.normalize = normalize
        
        # Functions
        self.functions = functions if functions is not None else ['add', 'sub', 'mul', 'div']
        self.condition_functions = condition_functions
        self.constant_range = constant_range
        
        # Selection
        self.selector = selector
        self.selector_params = params.get("selector_args", {})
        
        self.params = params
        self._is_fitted = False
    
    def fit(
        self,
        X,
        y,
        X_test=None,
        y_test=None,
        specialist_n_gen: int = 100,
        ensemble_n_gen: int = 100,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        """
        Fit the PSSR model.
        
        Phase 1: Train specialist population using GP
        Phase 2: Evolve ensemble trees using specialists
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        X_test : array-like, optional
            Test features for tracking test performance
        y_test : array-like, optional
            Test targets
        specialist_n_gen : int
            Number of generations for specialist training
        ensemble_n_gen : int
            Number of generations for ensemble evolution
        verbose : int
            Verbosity level (0: silent, 1: progress, 2: detailed)
        warm_start : bool
            If True, continue training from existing population
            
        Returns
        -------
        self
        """
        # Validate n_gen parameters
        if specialist_n_gen < 0:
            raise ValueError(f"specialist_n_gen must be >= 0, got {specialist_n_gen}")
        if ensemble_n_gen < 0:
            raise ValueError(f"ensemble_n_gen must be >= 0, got {ensemble_n_gen}")
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}"
            )
        
        # Normalize data
        X, y = self._fit_normalize(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Handle test data
        X_test_arr = None
        y_test_arr = None
        if X_test is not None and y_test is not None:
            X_test_arr = self._transform_X(np.asarray(X_test))
            if self.normalize and hasattr(self, 'y_scaler_') and self.y_scaler_ is not None:
                y_test_arr = self.y_scaler_.transform(
                    np.asarray(y_test).reshape(-1, 1)
                ).flatten()
            else:
                y_test_arr = np.asarray(y_test, dtype=float)
        
        # Initialize RNG
        self._rng_ = np.random.default_rng(self.random_state)
        
        # =====================================================================
        # Phase 1: Train Specialists
        # =====================================================================
        if verbose > 0:
            print("=" * 60)
            print("Phase 1: Training Specialists")
            print("=" * 60)
        
        gp_regressor = GPRegressor(
            population_size=self.specialist_pop_size,
            init_depth=self.specialist_init_depth,
            max_depth=self.specialist_max_depth,
            p_xo=self.p_xo,
            random_state=self.random_state,
            normalize=False,  # Already normalized
            functions=self.functions,
            constant_range=self.constant_range,
            selector=self.selector,
            **{k: v for k, v in self.params.items() if k not in ['selector_args']},
            selector_args=self.selector_params,
        )
        
        gp_regressor.fit(
            X, y,
            X_test=X_test_arr,
            y_test=y_test_arr,
            n_gen=specialist_n_gen,
            verbose=verbose,
        )
        
        self.specialist_population_ = gp_regressor.population_
        self.specialist_log_ = gp_regressor.log_
        
        if verbose > 0:
            print(f"\nSpecialist training complete.")
            print(f"Best specialist fitness: {gp_regressor.best_individual_.fitness:.6f}")
        
        # =====================================================================
        # Phase 2: Create Specialists and Build Ensemble Primitive Set
        # =====================================================================
        if verbose > 0:
            print("\n" + "=" * 60)
            print("Phase 2: Evolving Ensemble")
            print("=" * 60)
        
        # Build combined X and y for ensemble evaluation (train + test)
        if X_test_arr is not None:
            X_combined = np.vstack([X, X_test_arr])
            y_combined = np.concatenate([y, y_test_arr])
            train_slice = slice(0, X.shape[0])
            test_slice = slice(X.shape[0], X_combined.shape[0])
        else:
            X_combined = X
            y_combined = y
            train_slice = slice(0, X.shape[0])
            test_slice = None

        # Store slices for later use
        self._train_slice = train_slice
        self._test_slice = test_slice
        
        # Create specialists from population with combined semantics
        self.specialists_ = create_specialists_from_population(
            self.specialist_population_,
            X=X_combined,
        )
        
        if verbose > 0:
            print(f"Created {len(self.specialists_)} specialists")
        
        # Build ensemble primitive set
        # Use condition functions if specified, otherwise use same as specialists
        cond_functions = self.condition_functions
        if cond_functions is None:
            cond_functions = self.functions
        
        self.ensemble_primitive_set_ = PrimitiveSet(
            X=X_combined,
            functions=cond_functions,
            constant_range=self.constant_range,
        )
        self.ensemble_primitive_set_.set_specialists(self.specialists_)
        
        # =====================================================================
        # Phase 2: Initialize Ensemble Population
        # =====================================================================
        ensemble_population = ensemble_initializer(
            primitive_set=self.ensemble_primitive_set_,
            rng=self._rng_,
            population_size=self.ensemble_pop_size,
            depth_condition=self.depth_condition,
            max_depth=self.ensemble_max_depth,
        )
        
        self.ensemble_population_ = Population(ensemble_population)
        
        # =====================================================================
        # Phase 2: Create Variation Operators
        # =====================================================================
        crossover = fetch_ensemble_crossover("homologous", self.ensemble_max_depth)
        mutation = fetch_ensemble_mutation(
            primitive_set=self.ensemble_primitive_set_,
            rng=self._rng_,
            depth_condition=self.depth_condition,
            max_depth=self.ensemble_max_depth,
        )
        variator = ensemble_variator(crossover, mutation, self.p_xo)
        
        # Resolve selector
        if callable(self.selector):
            selector_ = self.selector
        else:
            selector_ = fetch_selector(self.selector, **self.selector_params)
        
        # =====================================================================
        # Phase 2: Ensemble Evolution Loop
        # =====================================================================
        self.ensemble_log_ = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
            "std_fitness": [],
            "best_depth": [],
            "best_nodes": [],
        }
        
        if X_test_arr is not None:
            self.ensemble_log_["best_test_fitness"] = []
            self.ensemble_log_["mean_test_fitness"] = []
        
        # Evaluate initial population using combined semantics
        self.ensemble_population_.set_slices(train_slice, test_slice)
        _calculate_ensemble_semantics_combined(
            self.ensemble_population_, train_slice, test_slice
        )
        self.ensemble_population_.calculate_errors_case(y_combined)
        self.ensemble_population_.evaluate(X_combined, y_combined)
        
        best_ind = self.ensemble_population_.get_best_individual()
        best_fitness = best_ind.fitness
        elites = self.ensemble_population_.extract_elites(1)
        
        self._log_generation(0, y_test_arr is not None)
        
        if verbose > 0:
            self._print_generation(0, first=True)
        
        # Main evolution loop
        for gen in range(1, ensemble_n_gen + 1):
            # Selection
            parents = self._apply_selector(selector_, self.ensemble_population_, self._rng_)
            
            # Variation
            offspring, _ = variator(
                parents.population,
                self.ensemble_primitive_set_,
                self._rng_,
                self.ensemble_max_depth,
            )
            
            # Create new population
            new_population = Population(offspring)
            
            # Inject elites
            if elites:
                new_population.inject_elites(elites, self._rng_)
            
            # Set slices and calculate combined semantics
            new_population.set_slices(train_slice, test_slice)
            _calculate_ensemble_semantics_combined(
                new_population, train_slice, test_slice
            )
            new_population.calculate_errors_case(y_combined)
            new_population.evaluate(X_combined, y_combined)
            
            self.ensemble_population_ = new_population
            
            # Track best
            gen_best = self.ensemble_population_.get_best_individual()
            if gen_best.fitness < best_fitness:
                best_fitness = gen_best.fitness
                best_ind = gen_best.copy()
            
            elites = self.ensemble_population_.extract_elites(1)
            
            self._log_generation(gen, y_test_arr is not None)
            
            if verbose > 0:
                self._print_generation(gen, first=False)
        
        self.best_ensemble_ = best_ind
        self._is_fitted = True
        
        if verbose > 0:
            print("\n" + "=" * 60)
            print("Training Complete")
            print(f"Best ensemble fitness: {best_fitness:.6f}")
            print("=" * 60)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the best ensemble.
        
        Parameters
        ----------
        X : array-like
            Input features
            
        Returns
        -------
        array
            Predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        X = self._transform_X(np.asarray(X))
        
        y_pred = self.best_ensemble_.predict(X, primitive_set=self.ensemble_primitive_set_)
        y_pred = self._inverse_transform_y(y_pred)
        
        return y_pred
    
    def get_specialists(self) -> dict[str, Specialist]:
        """Get the trained specialists."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.specialists_
    
    def get_best_ensemble(self) -> EnsembleIndividual:
        """Get the best ensemble individual."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.best_ensemble_
    
    def _apply_selector(
        self,
        selector: Callable,
        population: Population,
        rng: np.random.Generator,
    ) -> Population:
        """Apply selection to get parent population."""
        try:
            selection = selector(population, rng)
        except TypeError:
            selection = selector(population)
        
        if isinstance(selection, tuple):
            selection = selection[0]
        
        if hasattr(selection, 'population'):
            selected = selection.population
        else:
            selected = list(selection)
        
        return Population(selected)
    
    def _log_generation(self, generation: int, has_test: bool) -> None:
        """Log generation statistics."""
        pop = self.ensemble_population_
        
        self.ensemble_log_["generation"].append(generation)
        self.ensemble_log_["best_fitness"].append(float(np.min(pop.train_fitness)))
        self.ensemble_log_["mean_fitness"].append(float(np.mean(pop.train_fitness)))
        self.ensemble_log_["worst_fitness"].append(float(np.max(pop.train_fitness)))
        self.ensemble_log_["std_fitness"].append(float(np.std(pop.train_fitness)))
        
        best_ind = pop.get_best_individual()
        self.ensemble_log_["best_depth"].append(best_ind.depth)
        self.ensemble_log_["best_nodes"].append(best_ind.nodes_count)
        
        if has_test and pop.test_fitness is not None:
            self.ensemble_log_["best_test_fitness"].append(float(np.min(pop.test_fitness)))
            self.ensemble_log_["mean_test_fitness"].append(float(np.mean(pop.test_fitness)))
    
    def _print_generation(self, generation: int, first: bool) -> None:
        """Print generation progress."""
        pop = self.ensemble_population_
        best_fit = np.min(pop.train_fitness)
        mean_fit = np.mean(pop.train_fitness)
        best_ind = pop.get_best_individual()
        
        test_str = ""
        if pop.test_fitness is not None:
            test_str = f" | Test: {np.min(pop.test_fitness):.4f}"
        
        if first:
            print(f"Gen {generation:4d} | Train: {best_fit:.4f} (mean: {mean_fit:.4f})"
                  f"{test_str} | Depth: {best_ind.depth} | Nodes: {best_ind.nodes_count}")
        else:
            print(f"Gen {generation:4d} | Train: {best_fit:.4f} (mean: {mean_fit:.4f})"
                  f"{test_str} | Depth: {best_ind.depth} | Nodes: {best_ind.nodes_count}")
