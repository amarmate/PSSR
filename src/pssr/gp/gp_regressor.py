from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import RegressorMixin

from pssr.core.normalization import NormalizationMixin
from pssr.core.primitives import FunctionSet, PrimitiveSet
from pssr.core.representations.individual import Individual
from pssr.core.representations.population import Population
from pssr.core.selection import fetch_selector
from pssr.gp.gp_evolution import GPevo
from pssr.gp.gp_initialization import fetch_initializer
from pssr.gp.gp_presets import fetch_preset
from pssr.gp.gp_variation import fetch_crossover, fetch_mutation, variator_fun


class GPRegressor(NormalizationMixin, RegressorMixin):
    def __init__(self,
                 population_size: int = 100, 
                 init_depth: int = 2,
                 max_depth: int = 6, 
                 p_xo: float = 0.8, 
                 random_state: int = 42,
                 normalize: bool = False,
                 
                 preset: Optional[str] = None,
                 
                 functions: Optional[FunctionSet] = None,
                 constant_range : Optional[float] = None,
                 selector: Union[str, Callable] = "tournament",
                 initializer: Union[str, Callable] = "rhh",
                 crossover: Union[str, Callable] = "single_point",
                 mutation: Union[str, Callable] = "subtree",
                 
                 X_scaler = None,
                 y_scaler = None, 
                 **params) -> None:
        
        NormalizationMixin.__init__(self, normalize=normalize,
                                    X_scaler=X_scaler, y_scaler=y_scaler)
        
        # Validate parameters
        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}")
        if init_depth < 1:
            raise ValueError(f"init_depth must be >= 1, got {init_depth}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if init_depth > max_depth:
            raise ValueError(f"init_depth ({init_depth}) must be <= max_depth ({max_depth})")
        if not 0.0 <= p_xo <= 1.0:
            raise ValueError(f"p_xo must be in [0, 1], got {p_xo}")
        
        # basic configurations
        self.population_size    = population_size
        self.init_depth         = init_depth
        self.max_depth          = max_depth
        self.p_xo               = p_xo
        self.random_state       = random_state
        self.normalize          = normalize
        
        # advanced configurations
        self.preset             = preset
        self.constant_range    = constant_range
        self.functions          = functions
        self.selector           = selector
        self.initializer        = initializer
        self.crossover          = crossover
        self.mutation           = mutation
        self.params             = params
        
        # component-specific kwargs
        self.selector_params    = params.get("selector_args", {})
        self.initializer_params = params.get("initializer_args", {})
        self.crossover_params   = params.get("crossover_args", {})
        self.mutation_params    = params.get("mutation_args", {})
        
        self.params             = params 
        
        self._is_fitted = False
        self._n_generations_completed = 0  # Track total generations across warmstarts
    
    def fit(self, X, y, X_test=None, y_test=None, n_gen: int = 2000, verbose: int = 0, warm_start: bool = False): 
        if y is None:
            raise ValueError("y must be provided for GPRegressor.fit")
        
        if n_gen < 0:
            raise ValueError(f"n_gen must be >= 0, got {n_gen}")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}"
            )
        
        # Normalize training data first
        X, y = self._fit_normalize(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_train = X.shape[0]
        
        # Handle test data if provided
        if X_test is not None and y_test is not None:
            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test)
            
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(
                    f"X_test and y_test must have the same number of samples. "
                    f"Got X_test.shape[0]={X_test.shape[0]} and y_test.shape[0]={y_test.shape[0]}"
                )
            
            if X_test.shape[1] != X.shape[1]:
                raise ValueError(
                    f"X_test must have the same number of features as X. "
                    f"Got X.shape[1]={X.shape[1]} and X_test.shape[1]={X_test.shape[1]}"
                )
            
            # Normalize test data using training scalers (already fitted on train)
            X_test_arr = self._transform_X(X_test)
            # For y_test, use the same scaler as training
            if self.normalize and hasattr(self, 'y_scaler_') and self.y_scaler_ is not None:
                y_test_arr = self.y_scaler_.transform(y_test.reshape(-1, 1)).flatten()
            else:
                y_test_arr = np.asarray(y_test, dtype=float)
            
            # Merge train and test data
            X_combined = np.vstack([X, X_test_arr])
            y_combined = np.concatenate([y, y_test_arr])
            train_slice = slice(0, n_train)
            test_slice = slice(n_train, X_combined.shape[0])
        else:
            # No test data - use training data only
            X_combined = X
            y_combined = y
            train_slice = slice(0, n_train)
            test_slice = None
        
        # Warm start: continue from existing population
        if warm_start and self._is_fitted:
            # Validate that feature dimensions match
            if hasattr(self, 'primitive_set_') and self.primitive_set_ is not None:
                expected_n_features = len(self.primitive_set_.terminals_)
                if X.shape[1] != expected_n_features:
                    raise ValueError(
                        f"Cannot warm start with different number of features. "
                        f"Expected {expected_n_features}, got {X.shape[1]}"
                    )
            
            # Use existing components (don't reinitialize)
            if not hasattr(self, 'primitive_set_') or self.primitive_set_ is None:
                self._resolve_preset()
                self._resolve_primitive_set(X_combined)
            else:
                # Update primitive set with new X (and test X if available)
                # This is needed because primitive set stores X values internally
                self._resolve_primitive_set(X_combined)
            
            # Ensure RNG and components are initialized
            self._rng()
            if not hasattr(self, 'selector_'):
                self._resolve_selector()
            if not hasattr(self, 'variator_'):
                self._resolve_variator()
            
            # Use existing population
            population = self.population_
            
            # Re-evaluate population on new data
            population.train_semantics = None
            population.test_semantics = None
            population.train_fitness = None
            population.errors_case = None
            
            # Continue evolution
            population, best_individual, new_log = GPevo(
                population = population,
                X = X_combined,
                y = y_combined,
                primitive_set = self.primitive_set_,
                selector = self.selector_,
                variator = self.variator_,
                rng = self._rng_,
                n_generations = n_gen,
                max_depth = self.max_depth,
                train_slice = train_slice,
                test_slice = test_slice,
                verbose = verbose,
            )
            
            # Merge logs (append new generations to existing log)
            if hasattr(self, 'log_') and self.log_ is not None:
                # Adjust generation numbers in new log to continue from where we left off
                start_gen = self._n_generations_completed
                # Skip generation 0 from new log (it's a re-evaluation of current population)
                # and adjust remaining generation numbers
                skip_first = True
                for i, gen in enumerate(new_log["generation"]):
                    if skip_first and gen == 0:
                        skip_first = False
                        continue  # Skip generation 0 re-evaluation
                    adjusted_gen = start_gen + gen
                    self.log_["generation"].append(adjusted_gen)
                    self.log_["best_fitness"].append(new_log["best_fitness"][i])
                    self.log_["mean_fitness"].append(new_log["mean_fitness"][i])
                    self.log_["worst_fitness"].append(new_log["worst_fitness"][i])
                    self.log_["std_fitness"].append(new_log["std_fitness"][i])
                    self.log_["best_depth"].append(new_log["best_depth"][i])
                    self.log_["best_size"].append(new_log["best_size"][i])
                    self.log_["eval_time"].append(new_log["eval_time"][i])
                    # Add test fitness if available
                    if "best_test_fitness" in new_log and i < len(new_log["best_test_fitness"]):
                        if "best_test_fitness" not in self.log_:
                            # Initialize test fitness lists if not present
                            # Fill with 0.0 for previous generations (backfill)
                            prev_len = len(self.log_["generation"]) - 1
                            self.log_["best_test_fitness"] = [0.0] * prev_len
                            self.log_["mean_test_fitness"] = [0.0] * prev_len
                        self.log_["best_test_fitness"].append(new_log["best_test_fitness"][i])
                        self.log_["mean_test_fitness"].append(new_log["mean_test_fitness"][i])
            else:
                self.log_ = new_log
            
            self.population_ = population
            self.best_individual_ = best_individual
            self._n_generations_completed += n_gen
            
        else:
            # Fresh start: initialize new population
            self._resolve_preset()
            self._resolve_primitive_set(X_combined)
            
            self._rng()
            self._resolve_selector()
            self._resolve_variator()
            
            initializer_ = self._resolve_initializer()
            individuals = initializer_(
                self.primitive_set_,
                self._rng_,
                self.population_size,
                self.max_depth,
            )
            population = Population(individuals)
            
            population, best_individual, log = GPevo(
                population = population,
                X = X_combined,
                y = y_combined,
                primitive_set = self.primitive_set_,
                selector = self.selector_,
                variator = self.variator_,
                rng = self._rng_,
                n_generations = n_gen,
                max_depth = self.max_depth,
                train_slice = train_slice,
                test_slice = test_slice,
                verbose = verbose,
            )
            
            self.population_ = population
            self.best_individual_ = best_individual
            self.log_ = log
            self._n_generations_completed = n_gen
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")

        X = self._transform_X(X)

        best : Optional[Individual] = getattr(self, "best_individual_", None)
        if best is None:
            raise RuntimeError("Best individual not available. Fit the model first.")

        y_pred = best.predict(X)
        y_pred = self._inverse_transform_y(y_pred)
        return y_pred
    
    # ------------------------------------ # 
    # ----- Internal helper methods ------ # 
    # ------------------------------------ # 
 
    def _rng(self) -> None | np.random.Generator:
        if not hasattr(self, "_rng_"):
            self._rng_ = np.random.default_rng(self.random_state)
        return self._rng_
            
    def _resolve_preset(self) -> None:
        if self.preset is None:
            return
        
        preset          = fetch_preset(self.preset)
        self.preset_    = preset 

        if self.functions is None and "functions" in preset:
            self._preset_functions = preset["functions"]

        if self.constant_range is None and "constant_range" in preset:
            self._preset_constant_range = preset["constant_range"]

        if self.population_size == 100 and "population_size" in preset:
            self.population_size = preset["population_size"]

        # Note: n_gen is now passed to fit(), not stored in __init__

        if self.max_depth == 6 and "max_depth" in preset:
            self.max_depth = preset["max_depth"]
        
        if self.init_depth == 2 and "init_depth" in preset:
            self.init_depth = preset["init_depth"]
            
        if self.p_xo == 0.8 and "p_xo" in preset:
            self.p_xo = preset["p_xo"]

        if self.selector == "tournament" and "selector" in preset:
            self.selector = preset["selector"]

        if self.initializer == "rhh" and "initializer" in preset:
            self.initializer = preset["initializer"]

        if self.crossover == "single_point" and "crossover" in preset:
            self.crossover = preset["crossover"]

        if self.mutation == "subtree" and "mutation" in preset:
            self.mutation = preset["mutation"]
            
        if "selector_args" in preset:
            self._preset_selector_params = preset["selector_args"]

        if "initializer_args" in preset:
            self._preset_initializer_params = preset["initializer_args"]

        if "crossover_args" in preset:
            self._preset_crossover_params = preset["crossover_args"]

        if "mutation_args" in preset:
            self._preset_mutation_params = preset["mutation_args"]
            
    
    def _resolve_primitive_set(self, X: np.ndarray) -> None: 
        constant_range = self.constant_range
        if constant_range is None:
            constant_range = getattr(self, "_preset_constant_range", 1.0)
        
        functions = self.functions
        if functions is None:
            functions = getattr(self, "_preset_functions", ['add', 'sub', 'mul', 'div'])
        
        self.primitive_set_ = PrimitiveSet(
            X = X,
            functions=functions,
            constant_range=constant_range,
            )
    

    def _resolve_selector(self) -> Callable:
        if callable(self.selector):
            self.selector_ = self.selector
            return self.selector_
        
        preset_args = getattr(self, "_preset_selector_params", {})
        selector_kwargs = {**preset_args, **self.selector_params}

        self.selector_ =  fetch_selector(self.selector, **selector_kwargs)
        return self.selector_

    def _resolve_initializer(self) -> Callable:
        if callable(self.initializer):
            self.initializer_ = self.initializer
            return self.initializer_
        
        preset_args = getattr(self, "_preset_initializer_params", {})
        init_kwargs = {**preset_args, **self.initializer_params}

        self.initializer_ = fetch_initializer(
            init_depth=self.init_depth,
            max_depth=self.max_depth, 
            **init_kwargs)
        
        return self.initializer_
    
    def _resolve_variator(self) -> Callable:
        crossover = self._resolve_crossover()
        mutation = self._resolve_mutation()
        sample_by_level = self.params.get("mutation_sample_by_level", True)
        mutation_cache_size = self.params.get("mutation_cache_size", 5_000)
        
        self.variator_ = variator_fun(
            crossover=crossover,
            mutation=mutation,
            p_xo=self.p_xo,
            sample_by_level=sample_by_level,
            mutation_cache_size=mutation_cache_size,
        )
        return self.variator_
            
    def _resolve_crossover(self) -> Callable:
        if callable(self.crossover):
            self.crossover_ = self.crossover
            return self.crossover_
        
        preset_args = getattr(self, "_preset_crossover_params", {})
        xo_kwargs = {**preset_args, **self.crossover_params}
        
        self.crossover_ = fetch_crossover(self.crossover, **xo_kwargs)
        return self.crossover_

    def _resolve_mutation(self) -> Callable:
        if callable(self.mutation):
            self.mutation_ = self.mutation
            return self.mutation_
        
        preset_args = getattr(self, "_preset_mutation_params", {})
        mut_kwargs = {**preset_args, **self.mutation_params}
        
        self.mutation_ = fetch_mutation(self.mutation, **mut_kwargs)
        return self.mutation_
