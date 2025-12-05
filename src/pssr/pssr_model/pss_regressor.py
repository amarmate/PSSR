"""
PSSRegressor: Piecewise Specialist Symbolic Regression.

This module provides the main PSSRegressor class which implements
the Multi-SLIM approach for symbolic regression using piecewise
specialists with conditional routing.
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin

from pssr.core.fitness import fetch_fitness
from pssr.core.normalization import NormalizationMixin
from pssr.core.primitives import FunctionSet, PrimitiveSet
from pssr.core.representations.population import Population
from pssr.core.selection import fetch_selector
from pssr.gp.gp_evolution import GPevo
from pssr.gp.gp_initialization import fetch_initializer
from pssr.gp.gp_variation import fetch_crossover, fetch_mutation, variator_fun
from pssr.pssr_model.ensemble_individual import EnsembleIndividual
from pssr.pssr_model.ensemble_initialization import ensemble_initializer
from pssr.pssr_model.ensemble_operators import (
    ensemble_variator,
    fetch_ensemble_crossover,
    fetch_ensemble_mutation,
)
from pssr.pssr_model.pss_presets import fetch_pss_preset
from pssr.pssr_model.specialist import Specialist, create_specialists_from_population

if TYPE_CHECKING:
    from pssr.core.callbacks import Callback

Array = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


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
    preset : Optional[str]
        Name of a preset configuration. Overrides individual parameters.
        Available: "default", "small", "large", "lexicase"
    
    # Specialist training parameters
    specialist_pop_size : int
        Population size for specialist GP evolution
    specialist_max_depth : int
        Maximum tree depth for specialists
    specialist_init_depth : int
        Initial tree depth for specialists
    specialist_selector : Union[str, Callable]
        Selection method for specialist phase
    specialist_initializer : Union[str, Callable]
        Initialization method for specialists
    specialist_crossover : Union[str, Callable]
        Crossover method for specialists
    specialist_mutation : Union[str, Callable]
        Mutation method for specialists
    
    # Ensemble evolution parameters
    ensemble_pop_size : int
        Population size for ensemble evolution
    ensemble_max_depth : int
        Maximum depth for ensemble trees
    depth_condition : int
        Maximum depth for condition trees in ensemble
    ensemble_selector : Union[str, Callable]
        Selection method for ensemble phase
    ensemble_crossover : Union[str, Callable]
        Crossover method for ensemble (default: homologous)
    
    # Variation parameters
    p_xo : float
        Probability of crossover vs mutation (for both phases)
    
    # Elitism parameters
    specialist_n_elites : int
        Number of elite individuals to preserve in specialist phase
    ensemble_n_elites : int
        Number of elite individuals to preserve in ensemble phase
    
    # Callbacks
    specialist_callbacks : Optional[list[Callback]]
        Callbacks for the specialist training phase.
        Use callbacks for early stopping (e.g., EarlyStoppingCallback) or timeout logic.
    ensemble_callbacks : Optional[list[Callback]]
        Callbacks for the ensemble training phase.
        Use callbacks for early stopping (e.g., EarlyStoppingCallback) or timeout logic.
    
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
    fitness_function : Union[str, Callable]
        Fitness function name ("rmse", "mse", "mae", "r2") or callable.
        The callable should accept (y_true, y_pred) and return fitness values.
    
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
    specialist_log_ : dict
        Specialist training log
    ensemble_log_ : dict
        Ensemble training log
    """
    
    def __init__(
        self,
        # Preset
        preset: Optional[str] = None,
        
        # Specialist parameters
        specialist_pop_size: int = 100,
        specialist_max_depth: int = 6,
        specialist_init_depth: int = 2,
        specialist_selector: Union[str, Callable] = "dalex",
        specialist_initializer: Union[str, Callable] = "rhh",
        specialist_crossover: Union[str, Callable] = "single_point",
        specialist_mutation: Union[str, Callable] = "subtree",
        
        # Ensemble parameters
        ensemble_pop_size: int = 100,
        ensemble_max_depth: int = 4,
        depth_condition: int = 3,
        ensemble_selector: Union[str, Callable] = "tournament",
        ensemble_crossover: Union[str, Callable] = "homologous",
        
        # Variation parameters
        p_xo: float = 0.5,
        
        # Elitism parameters
        specialist_n_elites: int = 1,
        ensemble_n_elites: int = 1,
        
        # Callbacks
        specialist_callbacks: Optional[list["Callback"]] = None,
        ensemble_callbacks: Optional[list["Callback"]] = None,
        
        # General parameters
        random_state: int = 42,
        normalize: bool = False,
        
        # Function sets
        functions: Optional[Union[list[str], FunctionSet]] = None,
        condition_functions: Optional[Union[list[str], FunctionSet]] = None,
        constant_range: Optional[float] = None,
        
        # Fitness function
        fitness_function: Union[str, Callable[[Array, Array], Array]] = "rmse",
        
        # Scalers
        X_scaler=None,
        y_scaler=None,
        
        **params,
    ) -> None:
        
        NormalizationMixin.__init__(
            self, normalize=normalize, X_scaler=X_scaler, y_scaler=y_scaler
        )
        
        # Store preset for resolution
        self.preset = preset
        
        # Specialist parameters
        self.specialist_pop_size = specialist_pop_size
        self.specialist_max_depth = specialist_max_depth
        self.specialist_init_depth = specialist_init_depth
        self.specialist_selector = specialist_selector
        self.specialist_initializer = specialist_initializer
        self.specialist_crossover = specialist_crossover
        self.specialist_mutation = specialist_mutation
        
        # Ensemble parameters
        self.ensemble_pop_size = ensemble_pop_size
        self.ensemble_max_depth = ensemble_max_depth
        self.depth_condition = depth_condition
        self.ensemble_selector = ensemble_selector
        self.ensemble_crossover = ensemble_crossover
        
        # Variation
        self.p_xo = p_xo
        
        # Elitism
        if specialist_n_elites < 0:
            raise ValueError(f"specialist_n_elites must be >= 0, got {specialist_n_elites}")
        if ensemble_n_elites < 0:
            raise ValueError(f"ensemble_n_elites must be >= 0, got {ensemble_n_elites}")
        self.specialist_n_elites = specialist_n_elites
        self.ensemble_n_elites = ensemble_n_elites
        
        # Callbacks
        self.specialist_callbacks = specialist_callbacks
        self.ensemble_callbacks = ensemble_callbacks
        
        # General
        self.random_state = random_state
        self.normalize = normalize
        
        # Functions
        self.functions = functions
        self.condition_functions = condition_functions
        self.constant_range = constant_range
        
        # Fitness function
        self.fitness_function = fitness_function
        
        # Component-specific kwargs
        self.params = params
        self.specialist_selector_params = params.get("specialist_selector_args", {})
        self.specialist_initializer_params = params.get("specialist_initializer_args", {})
        self.specialist_crossover_params = params.get("specialist_crossover_args", {})
        self.specialist_mutation_params = params.get("specialist_mutation_args", {})
        self.ensemble_selector_params = params.get("ensemble_selector_args", {})
        
        # State tracking
        self._is_fitted = False
        self._specialist_generations_completed = 0
        self._ensemble_generations_completed = 0
    
    def fit(
        self,
        X,
        y,
        X_test=None,
        y_test=None,
        specialist_n_gen: int = 100,
        ensemble_n_gen: int = 100,
        verbose: Union[int, str] = 0,
        warm_start: bool = False,
        warm_start_mode: str = "full",
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
        verbose : Union[int, str]
            Verbosity level:
            - 0: Silent
            - 1: Print every generation
            - N (int > 1): Print every N generations
            - "bar": Show tqdm progress bar
        warm_start : bool
            If True, continue training from existing population
        warm_start_mode : str
            Mode for warm start:
            - "full": Continue specialist evolution, then rerun ensemble
            - "ensemble_only": Keep specialists fixed, only rerun ensemble
            
        Returns
        -------
        self
        """
        # Validate n_gen parameters
        if specialist_n_gen < 0:
            raise ValueError(f"specialist_n_gen must be >= 0, got {specialist_n_gen}")
        if ensemble_n_gen < 0:
            raise ValueError(f"ensemble_n_gen must be >= 0, got {ensemble_n_gen}")
        if warm_start_mode not in ("full", "ensemble_only"):
            raise ValueError(
                f"warm_start_mode must be 'full' or 'ensemble_only', got {warm_start_mode}"
            )
        
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
        X_test_arr: Optional[Array] = None
        y_test_arr: Optional[Array] = None
        if X_test is not None and y_test is not None:
            X_test_arr = self._transform_X(np.asarray(X_test))
            if self.normalize and hasattr(self, 'y_scaler_') and self.y_scaler_ is not None:
                y_test_arr = self.y_scaler_.transform(
                    np.asarray(y_test).reshape(-1, 1)
                ).flatten()
            else:
                y_test_arr = np.asarray(y_test, dtype=float)
        
        # Resolve preset and components
        self._resolve_preset()
        self._validate_parameters()
        
        # Initialize RNG
        self._rng()
        
        # Check if verbose is "truthy" for logging (but not "bar" mode for header messages)
        verbose_log = verbose and verbose != "bar"
        
        # Handle warm start
        if warm_start and self._is_fitted:
            if warm_start_mode == "full":
                # Continue specialist evolution, then rerun ensemble
                if verbose_log:
                    logger.info("=" * 60)
                    logger.info("Warm Start: Continuing Specialist Evolution")
                    logger.info("=" * 60)
                
                self._warm_start_specialists(
                    X, y, X_test_arr, y_test_arr, specialist_n_gen, verbose
                )
                
                # Rerun ensemble from scratch with updated specialists
                self._fit_ensemble(X, y, X_test_arr, y_test_arr, ensemble_n_gen, verbose)
                
            else:  # ensemble_only
                # Keep specialists fixed, only rerun ensemble
                if verbose_log:
                    logger.info("=" * 60)
                    logger.info("Warm Start: Ensemble Only (specialists fixed)")
                    logger.info("=" * 60)
                
                self._warm_start_ensemble(
                    X, y, X_test_arr, y_test_arr, ensemble_n_gen, verbose
                )
        else:
            # Fresh start
            self._fit_specialists(X, y, X_test_arr, y_test_arr, specialist_n_gen, verbose)
            self._fit_ensemble(X, y, X_test_arr, y_test_arr, ensemble_n_gen, verbose)
        
        self._is_fitted = True
        
        if verbose_log:
            logger.info("\n" + "=" * 60)
            logger.info("Training Complete")
            logger.info(f"Best ensemble fitness: {self.best_ensemble_.fitness:.6f}")
            logger.info("=" * 60)
        
        return self
    
    # =========================================================================
    # Internal Resolution Methods
    # =========================================================================
    
    def _rng(self) -> np.random.Generator:
        """Get or create the random number generator."""
        if not hasattr(self, "_rng_"):
            self._rng_ = np.random.default_rng(self.random_state)
        return self._rng_
    
    def _resolve_preset(self) -> None:
        """Apply preset configuration if specified."""
        if self.preset is None:
            return
        
        preset = fetch_pss_preset(self.preset)
        self._preset_ = preset
        
        # Shared parameters
        if self.functions is None and "functions" in preset:
            self._preset_functions = preset["functions"]
        if self.constant_range is None and "constant_range" in preset:
            self._preset_constant_range = preset["constant_range"]
        if self.p_xo == 0.5 and "p_xo" in preset:
            self.p_xo = preset["p_xo"]
        
        # Specialist parameters
        if self.specialist_pop_size == 100 and "specialist_pop_size" in preset:
            self.specialist_pop_size = preset["specialist_pop_size"]
        if self.specialist_max_depth == 6 and "specialist_max_depth" in preset:
            self.specialist_max_depth = preset["specialist_max_depth"]
        if self.specialist_init_depth == 2 and "specialist_init_depth" in preset:
            self.specialist_init_depth = preset["specialist_init_depth"]
        if self.specialist_selector == "tournament" and "specialist_selector" in preset:
            self.specialist_selector = preset["specialist_selector"]
        if self.specialist_initializer == "rhh" and "specialist_initializer" in preset:
            self.specialist_initializer = preset["specialist_initializer"]
        if self.specialist_crossover == "single_point" and "specialist_crossover" in preset:
            self.specialist_crossover = preset["specialist_crossover"]
        if self.specialist_mutation == "subtree" and "specialist_mutation" in preset:
            self.specialist_mutation = preset["specialist_mutation"]
        
        # Specialist component args from preset
        if "specialist_selector_args" in preset:
            self._preset_specialist_selector_params = preset["specialist_selector_args"]
        if "specialist_initializer_args" in preset:
            self._preset_specialist_initializer_params = preset["specialist_initializer_args"]
        if "specialist_crossover_args" in preset:
            self._preset_specialist_crossover_params = preset["specialist_crossover_args"]
        if "specialist_mutation_args" in preset:
            self._preset_specialist_mutation_params = preset["specialist_mutation_args"]
        
        # Ensemble parameters
        if self.ensemble_pop_size == 100 and "ensemble_pop_size" in preset:
            self.ensemble_pop_size = preset["ensemble_pop_size"]
        if self.ensemble_max_depth == 4 and "ensemble_max_depth" in preset:
            self.ensemble_max_depth = preset["ensemble_max_depth"]
        if self.depth_condition == 3 and "depth_condition" in preset:
            self.depth_condition = preset["depth_condition"]
        if self.ensemble_selector == "tournament" and "ensemble_selector" in preset:
            self.ensemble_selector = preset["ensemble_selector"]
        if self.ensemble_crossover == "homologous" and "ensemble_crossover" in preset:
            self.ensemble_crossover = preset["ensemble_crossover"]
        
        # Ensemble component args from preset
        if "ensemble_selector_args" in preset:
            self._preset_ensemble_selector_params = preset["ensemble_selector_args"]
        
        # Condition functions
        if self.condition_functions is None and "condition_functions" in preset:
            self._preset_condition_functions = preset["condition_functions"]
    
    def _validate_parameters(self) -> None:
        """Validate all parameters after preset resolution."""
        if self.specialist_pop_size < 1:
            raise ValueError(f"specialist_pop_size must be >= 1, got {self.specialist_pop_size}")
        if self.ensemble_pop_size < 1:
            raise ValueError(f"ensemble_pop_size must be >= 1, got {self.ensemble_pop_size}")
        if self.specialist_init_depth < 1:
            raise ValueError(f"specialist_init_depth must be >= 1, got {self.specialist_init_depth}")
        if self.specialist_max_depth < 1:
            raise ValueError(f"specialist_max_depth must be >= 1, got {self.specialist_max_depth}")
        if self.specialist_init_depth > self.specialist_max_depth:
            raise ValueError(
                f"specialist_init_depth ({self.specialist_init_depth}) must be <= "
                f"specialist_max_depth ({self.specialist_max_depth})"
            )
        if not 0.0 <= self.p_xo <= 1.0:
            raise ValueError(f"p_xo must be in [0, 1], got {self.p_xo}")
    
    def _get_functions(self) -> Union[list[str], FunctionSet]:
        """Get functions with preset fallback."""
        if self.functions is not None:
            return self.functions
        return getattr(self, "_preset_functions", ["add", "sub", "mul", "div"])
    
    def _get_constant_range(self) -> float:
        """Get constant_range with preset fallback."""
        if self.constant_range is not None:
            return self.constant_range
        return getattr(self, "_preset_constant_range", 1.0)
    
    def _get_condition_functions(self) -> Union[list[str], FunctionSet]:
        """Get condition functions with fallback to regular functions."""
        if self.condition_functions is not None:
            return self.condition_functions
        if hasattr(self, "_preset_condition_functions"):
            return self._preset_condition_functions
        return self._get_functions()
    
    def _resolve_fitness_function(self) -> Callable[[Array, Array], Array]:
        """Resolve the fitness function from string name or callable."""
        if callable(self.fitness_function):
            return self.fitness_function
        return fetch_fitness(self.fitness_function)
    
    def _resolve_specialist_primitive_set(self, X: np.ndarray) -> None:
        """Create primitive set for specialist phase."""
        self.specialist_primitive_set_ = PrimitiveSet(
            X=X,
            functions=self._get_functions(),
            constant_range=self._get_constant_range(),
        )
    
    def _resolve_ensemble_primitive_set(self, X_combined: np.ndarray) -> None:
        """Create primitive set for ensemble phase."""
        self.ensemble_primitive_set_ = PrimitiveSet(
            X=X_combined,
            functions=self._get_condition_functions(),
            constant_range=self._get_constant_range(),
        )
        self.ensemble_primitive_set_.set_specialists(self.specialists_)
    
    def _resolve_specialist_selector(self) -> Callable:
        """Resolve specialist selector."""
        if callable(self.specialist_selector):
            self.specialist_selector_ = self.specialist_selector
            return self.specialist_selector_
        
        preset_args = getattr(self, "_preset_specialist_selector_params", {})
        selector_kwargs = {**preset_args, **self.specialist_selector_params}
        
        self.specialist_selector_ = fetch_selector(self.specialist_selector, **selector_kwargs)
        return self.specialist_selector_
    
    def _resolve_ensemble_selector(self) -> Callable:
        """Resolve ensemble selector."""
        if callable(self.ensemble_selector):
            self.ensemble_selector_ = self.ensemble_selector
            return self.ensemble_selector_
        
        preset_args = getattr(self, "_preset_ensemble_selector_params", {})
        selector_kwargs = {**preset_args, **self.ensemble_selector_params}
        
        self.ensemble_selector_ = fetch_selector(self.ensemble_selector, **selector_kwargs)
        return self.ensemble_selector_
    
    def _resolve_specialist_initializer(self) -> Callable:
        """Resolve specialist initializer."""
        if callable(self.specialist_initializer):
            self.specialist_initializer_ = self.specialist_initializer
            return self.specialist_initializer_
        
        preset_args = getattr(self, "_preset_specialist_initializer_params", {})
        init_kwargs = {**preset_args, **self.specialist_initializer_params}
        
        self.specialist_initializer_ = fetch_initializer(
            init_depth=self.specialist_init_depth,
            max_depth=self.specialist_max_depth,
            **init_kwargs,
        )
        return self.specialist_initializer_
    
    def _resolve_specialist_crossover(self) -> Callable:
        """Resolve specialist crossover."""
        if callable(self.specialist_crossover):
            self.specialist_crossover_ = self.specialist_crossover
            return self.specialist_crossover_
        
        preset_args = getattr(self, "_preset_specialist_crossover_params", {})
        xo_kwargs = {**preset_args, **self.specialist_crossover_params}
        
        self.specialist_crossover_ = fetch_crossover(self.specialist_crossover, **xo_kwargs)
        return self.specialist_crossover_
    
    def _resolve_specialist_mutation(self) -> Callable:
        """Resolve specialist mutation."""
        if callable(self.specialist_mutation):
            self.specialist_mutation_ = self.specialist_mutation
            return self.specialist_mutation_
        
        preset_args = getattr(self, "_preset_specialist_mutation_params", {})
        mut_kwargs = {**preset_args, **self.specialist_mutation_params}
        
        self.specialist_mutation_ = fetch_mutation(self.specialist_mutation, **mut_kwargs)
        return self.specialist_mutation_
    
    def _resolve_specialist_variator(self) -> Callable:
        """Resolve specialist variator combining crossover and mutation."""
        crossover = self._resolve_specialist_crossover()
        mutation = self._resolve_specialist_mutation()
        
        sample_by_level = self.params.get("specialist_mutation_sample_by_level", True)
        mutation_cache_size = self.params.get("specialist_mutation_cache_size", 5_000)
        
        self.specialist_variator_ = variator_fun(
            crossover=crossover,
            mutation=mutation,
            p_xo=self.p_xo,
            sample_by_level=sample_by_level,
            mutation_cache_size=mutation_cache_size,
        )
        return self.specialist_variator_
    
    def _resolve_ensemble_variator(self) -> Callable:
        """Resolve ensemble variator combining crossover and mutation."""
        # Ensemble crossover
        if callable(self.ensemble_crossover):
            crossover = self.ensemble_crossover
        else:
            crossover = fetch_ensemble_crossover(
                self.ensemble_crossover, 
                self.ensemble_max_depth
            )
        
        # Ensemble mutation (always uses factory)
        mutation = fetch_ensemble_mutation(
            primitive_set=self.ensemble_primitive_set_,
            rng=self._rng_,
            depth_condition=self.depth_condition,
            max_depth=self.ensemble_max_depth,
        )
        
        self.ensemble_variator_ = ensemble_variator(crossover, mutation, self.p_xo)
        return self.ensemble_variator_
    
    # =========================================================================
    # Training Methods
    # =========================================================================
    
    def _fit_specialists(
        self,
        X: Array,
        y: Array,
        X_test: Optional[Array],
        y_test: Optional[Array],
        n_gen: int,
        verbose: Union[int, str],
    ) -> None:
        """
        Phase 1: Train specialist population using GPevo directly.
        
        Parameters
        ----------
        X : Array
            Training features (normalized)
        y : Array
            Training targets (normalized)
        X_test : Optional[Array]
            Test features (normalized)
        y_test : Optional[Array]
            Test targets (normalized)
        n_gen : int
            Number of generations for specialist training
        verbose : int
            Verbosity level
        """
        verbose_log = verbose and verbose != "bar"
        if verbose_log:
            logger.info("=" * 60)
            logger.info("Phase 1: Training Specialists")
            logger.info("=" * 60)
        
        # Build combined data
        n_train = X.shape[0]
        if X_test is not None and y_test is not None:
            X_combined = np.vstack([X, X_test])
            y_combined = np.concatenate([y, y_test])
            train_slice = slice(0, n_train)
            test_slice = slice(n_train, X_combined.shape[0])
        else:
            X_combined = X
            y_combined = y
            train_slice = slice(0, n_train)
            test_slice = None
        
        # Resolve components
        self._resolve_specialist_primitive_set(X_combined)
        self._resolve_specialist_selector()
        self._resolve_specialist_variator()
        
        # Initialize population
        initializer = self._resolve_specialist_initializer()
        individuals = initializer(
            self.specialist_primitive_set_,
            self._rng_,
            self.specialist_pop_size,
            self.specialist_max_depth,
        )
        population = Population(individuals)
        
        # Run evolution
        population, best_individual, log = GPevo(
            population=population,
            X=X_combined,
            y=y_combined,
            primitive_set=self.specialist_primitive_set_,
            selector=self.specialist_selector_,
            variator=self.specialist_variator_,
            rng=self._rng_,
            n_generations=n_gen,
            max_depth=self.specialist_max_depth,
            train_slice=train_slice,
            test_slice=test_slice,
            elitism=self.specialist_n_elites,
            verbose=verbose,
            fitness_function=self.fitness_function,  # Pass original (string or callable)
            callbacks=self.specialist_callbacks,
        )
        
        self.specialist_population_ = population
        self.best_specialist_ = best_individual
        self.specialist_log_ = log
        self._specialist_generations_completed = n_gen
        
        if verbose_log:
            logger.info("\nSpecialist training complete.")
            logger.info(f"Best specialist fitness: {best_individual.fitness:.6f}")
    
    def _warm_start_specialists(
        self,
        X: Array,
        y: Array,
        X_test: Optional[Array],
        y_test: Optional[Array],
        n_gen: int,
        verbose: Union[int, str],
    ) -> None:
        """Continue specialist evolution from existing population."""
        # Build combined data
        n_train = X.shape[0]
        if X_test is not None and y_test is not None:
            X_combined = np.vstack([X, X_test])
            y_combined = np.concatenate([y, y_test])
            train_slice = slice(0, n_train)
            test_slice = slice(n_train, X_combined.shape[0])
        else:
            X_combined = X
            y_combined = y
            train_slice = slice(0, n_train)
            test_slice = None
        
        # Update primitive set with new data
        self._resolve_specialist_primitive_set(X_combined)
        
        # Ensure components are resolved
        if not hasattr(self, "specialist_selector_"):
            self._resolve_specialist_selector()
        if not hasattr(self, "specialist_variator_"):
            self._resolve_specialist_variator()
        
        # Use existing population
        population = self.specialist_population_
        
        # Clear cached values
        population.train_semantics = None
        population.test_semantics = None
        population.train_fitness = None
        population.errors_case = None
        
        # Continue evolution
        population, best_individual, new_log = GPevo(
            population=population,
            X=X_combined,
            y=y_combined,
            primitive_set=self.specialist_primitive_set_,
            selector=self.specialist_selector_,
            variator=self.specialist_variator_,
            rng=self._rng_,
            n_generations=n_gen,
            max_depth=self.specialist_max_depth,
            train_slice=train_slice,
            test_slice=test_slice,
            elitism=self.specialist_n_elites,
            verbose=verbose,
            fitness_function=self.fitness_function,  # Pass original (string or callable)
            callbacks=self.specialist_callbacks,
        )
        
        # Merge logs
        self._merge_logs(self.specialist_log_, new_log, self._specialist_generations_completed)
        
        self.specialist_population_ = population
        self.best_specialist_ = best_individual
        self._specialist_generations_completed += n_gen
        
        verbose_log = verbose and verbose != "bar"
        if verbose_log:
            logger.info("\nSpecialist warm start complete.")
            logger.info(f"Best specialist fitness: {best_individual.fitness:.6f}")
    
    def _fit_ensemble(
        self,
        X: Array,
        y: Array,
        X_test: Optional[Array],
        y_test: Optional[Array],
        n_gen: int,
        verbose: Union[int, str],
    ) -> None:
        """
        Phase 2: Evolve ensemble trees using GPevo.
        
        Parameters
        ----------
        X : Array
            Training features (normalized)
        y : Array
            Training targets (normalized)
        X_test : Optional[Array]
            Test features (normalized)
        y_test : Optional[Array]
            Test targets (normalized)
        n_gen : int
            Number of generations for ensemble evolution
        verbose : int
            Verbosity level
        """
        verbose_log = verbose and verbose != "bar"
        if verbose_log:
            logger.info("\n" + "=" * 60)
            logger.info("Phase 2: Evolving Ensemble")
            logger.info("=" * 60)
        
        # Build combined X and y for ensemble evaluation
        if X_test is not None and y_test is not None:
            X_combined = np.vstack([X, X_test])
            y_combined = np.concatenate([y, y_test])
            train_slice = slice(0, X.shape[0])
            test_slice = slice(X.shape[0], X_combined.shape[0])
        else:
            X_combined = X
            y_combined = y
            train_slice = slice(0, X.shape[0])
            test_slice = None
        
        # Store slices
        self._train_slice = train_slice
        self._test_slice = test_slice
        
        # Create specialists from population
        self.specialists_ = create_specialists_from_population(
            self.specialist_population_,
            X=X_combined,
        )
        
        if verbose_log:
            logger.info(f"Created {len(self.specialists_)} specialists")
        
        # Build ensemble primitive set
        self._resolve_ensemble_primitive_set(X_combined)
        
        # Initialize ensemble population
        ensemble_individuals = ensemble_initializer(
            primitive_set=self.ensemble_primitive_set_,
            rng=self._rng_,
            population_size=self.ensemble_pop_size,
            depth_condition=self.depth_condition,
            max_depth=self.ensemble_max_depth,
        )
        self.ensemble_population_ = Population(ensemble_individuals)
        
        # Resolve components
        self._resolve_ensemble_selector()
        self._resolve_ensemble_variator()
        
        # Run evolution
        self.ensemble_population_, self.best_ensemble_, self.ensemble_log_ = GPevo(
            population=self.ensemble_population_,
            X=X_combined,
            y=y_combined,
            primitive_set=self.ensemble_primitive_set_,
            selector=self.ensemble_selector_,
            variator=self.ensemble_variator_,
            rng=self._rng_,
            n_generations=n_gen,
            max_depth=self.ensemble_max_depth,
            train_slice=train_slice,
            test_slice=test_slice,
            elitism=self.ensemble_n_elites,
            verbose=verbose,
            fitness_function=self.fitness_function,  # Pass original (string or callable)
            callbacks=self.ensemble_callbacks,
        )
        
        self._ensemble_generations_completed = n_gen
    
    def _warm_start_ensemble(
        self,
        X: Array,
        y: Array,
        X_test: Optional[Array],
        y_test: Optional[Array],
        n_gen: int,
        verbose: Union[int, str],
    ) -> None:
        """Continue ensemble evolution from existing population."""
        # Build combined data
        if X_test is not None and y_test is not None:
            X_combined = np.vstack([X, X_test])
            y_combined = np.concatenate([y, y_test])
            train_slice = slice(0, X.shape[0])
            test_slice = slice(X.shape[0], X_combined.shape[0])
        else:
            X_combined = X
            y_combined = y
            train_slice = slice(0, X.shape[0])
            test_slice = None
        
        # Store slices
        self._train_slice = train_slice
        self._test_slice = test_slice
        
        # Recreate specialists with new data (semantics need updating)
        self.specialists_ = create_specialists_from_population(
            self.specialist_population_,
            X=X_combined,
        )
        
        verbose_log = verbose and verbose != "bar"
        if verbose_log:
            logger.info(f"Using {len(self.specialists_)} existing specialists")
        
        # Update ensemble primitive set
        self._resolve_ensemble_primitive_set(X_combined)
        
        # Ensure components are resolved
        if not hasattr(self, "ensemble_selector_"):
            self._resolve_ensemble_selector()
        
        # Re-resolve variator with updated primitive set
        self._resolve_ensemble_variator()
        
        # Use existing ensemble population
        population = self.ensemble_population_
        
        # Clear cached values
        population.train_semantics = None
        population.test_semantics = None
        population.train_fitness = None
        population.errors_case = None
        
        # Continue evolution
        population, best_ensemble, new_log = GPevo(
            population=population,
            X=X_combined,
            y=y_combined,
            primitive_set=self.ensemble_primitive_set_,
            selector=self.ensemble_selector_,
            variator=self.ensemble_variator_,
            rng=self._rng_,
            n_generations=n_gen,
            max_depth=self.ensemble_max_depth,
            train_slice=train_slice,
            test_slice=test_slice,
            elitism=self.ensemble_n_elites,
            verbose=verbose,
            fitness_function=self.fitness_function,  # Pass original (string or callable)
            callbacks=self.ensemble_callbacks,
        )
        
        # Merge logs
        self._merge_logs(self.ensemble_log_, new_log, self._ensemble_generations_completed)
        
        self.ensemble_population_ = population
        self.best_ensemble_ = best_ensemble
        self._ensemble_generations_completed += n_gen
        
        if verbose_log:
            logger.info("\nEnsemble warm start complete.")
            logger.info(f"Best ensemble fitness: {best_ensemble.fitness:.6f}")
    
    def _merge_logs(
        self, 
        existing_log: dict, 
        new_log: dict, 
        start_gen: int
    ) -> None:
        """Merge new log entries into existing log."""
        # Skip generation 0 from new log (re-evaluation) and adjust generation numbers
        skip_first = True
        for i, gen in enumerate(new_log["generation"]):
            if skip_first and gen == 0:
                skip_first = False
                continue
            
            adjusted_gen = start_gen + gen
            existing_log["generation"].append(adjusted_gen)
            existing_log["best_fitness"].append(new_log["best_fitness"][i])
            existing_log["mean_fitness"].append(new_log["mean_fitness"][i])
            existing_log["worst_fitness"].append(new_log["worst_fitness"][i])
            existing_log["std_fitness"].append(new_log["std_fitness"][i])
            existing_log["best_depth"].append(new_log["best_depth"][i])
            existing_log["best_size"].append(new_log["best_size"][i])
            existing_log["eval_time"].append(new_log["eval_time"][i])
            
            # Add test fitness if available
            if "best_test_fitness" in new_log and i < len(new_log["best_test_fitness"]):
                if "best_test_fitness" not in existing_log:
                    prev_len = len(existing_log["generation"]) - 1
                    existing_log["best_test_fitness"] = [0.0] * prev_len
                    existing_log["mean_test_fitness"] = [0.0] * prev_len
                existing_log["best_test_fitness"].append(new_log["best_test_fitness"][i])
                existing_log["mean_test_fitness"].append(new_log["mean_test_fitness"][i])
    
    # =========================================================================
    # Prediction and Accessor Methods
    # =========================================================================
    
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
    
    def get_specialist_generations(self) -> int:
        """Get total specialist generations completed."""
        return self._specialist_generations_completed
    
    def get_ensemble_generations(self) -> int:
        """Get total ensemble generations completed."""
        return self._ensemble_generations_completed
