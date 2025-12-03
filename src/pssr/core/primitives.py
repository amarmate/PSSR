from dataclasses import dataclass
from typing import Callable, Union, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .functions import DEFAULT_FUNCTIONS, GPFunction

if TYPE_CHECKING:
    from pssr.pssr_model.specialist import Specialist

Array = npt.NDArray[np.float64]

class FunctionSet:
    """Container for GP (Genetic Programming) functions with a registry and extensibility.

    This class stores a collection of callable functions used in GP evaluation.
    Each function is a ``GPFunction`` instance that defines:
        - a name,
        - a NumPy-compatible callable,
        - an arity (number of arguments).

    A ``FunctionSet`` can be created in multiple ways:
        - by passing a list of ``GPFunction`` objects,
        - by selecting function names from predefined defaults,
        - by constructing the default full set,
        - by registering additional user-defined functions.

    The container guarantees:
        - fast name-based lookup,
        - NumPy-safe wrapping of custom functions,
        - flexible overwrite behavior,
        - easy extension for advanced GP use cases.

    Example:
        >>> fs = FunctionSet.from_names(["add", "mul"])
        >>> def my_pow(x, y):
        ...     return x ** y
        >>> fs.add_function("pow", my_pow, arity=2)
        >>> f = fs.get("pow")
        >>> import numpy as np
        >>> f.func(np.array([2, 3]), np.array([3, 2]))
        array([8., 9.])
    """

    def __init__(self, functions: list[GPFunction]) -> None:
        """Initialize the FunctionSet with a list of GPFunction instances.

        Args:
            functions (list[GPFunction]): Functions to populate the registry.
        """
        self._functions_by_name: dict[str, GPFunction] = {
            f.name: f for f in functions
        }

    @classmethod
    def from_names(cls, names: list[str]) -> "FunctionSet":
        """Construct a FunctionSet using names of predefined default functions.

        Args:
            names (list[str]): Names of functions to include. Must exist in
                ``DEFAULT_FUNCTIONS``.

        Returns:
            FunctionSet: A function set including only the listed functions.

        Raises:
            ValueError: If any requested name does not exist in ``DEFAULT_FUNCTIONS``.

        Example:
            >>> fs = FunctionSet.from_names(["add", "sub"])
            >>> [f.name for f in fs.functions]
            ['add', 'sub']
        """
        funcs: list[GPFunction] = []
        for name in names:
            if name not in DEFAULT_FUNCTIONS:
                raise ValueError(f"Unknown function name: {name!r}")
            funcs.append(DEFAULT_FUNCTIONS[name])
        return cls(funcs)

    @classmethod
    def default(cls) -> "FunctionSet":
        """Create a FunctionSet containing all predefined default functions.

        Returns:
            FunctionSet: The complete default function set.

        Example:
            >>> fs = FunctionSet.default()
            >>> sorted(f.name for f in fs.functions)
            ['add', 'aq', 'cos', 'div', 'exp', 'log', 'mul', 'neg', 'sin', 'sqrt', 'sub']
        """
        return cls(list(DEFAULT_FUNCTIONS.values()))

    def add_function(
        self,
        name: str,
        func: Callable[..., Array],
        arity: int,
        overwrite: bool = False,
    ) -> None:
        """Register a new function in the set.

        The function is automatically wrapped to ensure NumPy-array compatibility.

        Args:
            name (str): Name of the function. Must be unique unless ``overwrite=True``.
            func (Callable[..., Array]): A callable taking NumPy arrays as input.
            arity (int): Number of arguments the function expects.
            overwrite (bool): Whether to override an existing function. Defaults to False.

        Raises:
            ValueError: If the function already exists and overwrite=False.

        Example:
            >>> fs = FunctionSet.from_names(["add"])
            >>> def diff_abs(x, y):
            ...     return np.abs(x - y)
            >>> fs.add_function("absdiff", diff_abs, arity=2)
            >>> fs.get("absdiff").arity
            2
        """
        if not overwrite and name in self._functions_by_name:
            raise ValueError(
                f"Function {name!r} already exists. Use overwrite=True to replace it."
            )

        def wrapped(*args):
            np_args = [np.asarray(a, dtype=float) for a in args]
            return np.asarray(func(*np_args), dtype=float)

        self._functions_by_name[name] = GPFunction(
            name=name,
            func=wrapped,
            arity=arity,
        )

    @property
    def functions(self) -> list[GPFunction]:
        """List all GP functions registered in this set.

        Returns:
            list[GPFunction]: All stored functions.

        Example:
            >>> fs = FunctionSet.from_names(["add"])
            >>> len(fs.functions)
            1
        """
        return list(self._functions_by_name.values())

    def get(self, name: str) -> GPFunction:
        """Retrieve a function by its name.

        Args:
            name (str): Name of the function.

        Returns:
            GPFunction: The corresponding function object.

        Raises:
            KeyError: If the function name is not registered.

        Example:
            >>> fs = FunctionSet.from_names(["mul"])
            >>> f = fs.get("mul")
            >>> f.name
            'mul'
        """
        return self._functions_by_name[name]
    
@dataclass(frozen=True)
class GPConstant:
    """
    Represents a Genetic Programming constant with a name and associated array.
    """
    name: str
    values: Array
    
@dataclass(frozen=True)
class GPTerminal:
    """
    Represents a Genetic Programming terminal (variable) with a name and associated array.
    """
    name: str
    values: Array


class PrimitiveSet:
    """Primitive set for GP trees, including functions, terminals and constants.

    This class ties together:
      * a set of functions (via ``FunctionSet`` or a list of function names),
      * input terminals derived from the feature matrix ``X``,
      * a grid of constant terminals in a symmetric numeric range.

    Terminals are created as:
      * ``x0, x1, ..., x{n_features-1}`` for each input column of ``X``
      * ``c0, c1, ..., c{n_constants-1}`` for evenly spaced constants in
        ``[-constant_range, constant_range]``.

    The primitive set is intended to be used by the GP engine during tree
    creation and evaluation.

    Args:
        X (np.ndarray): Input data of shape ``(n_samples, n_features)`` used
            to define the terminal symbols ``x0, x1, ...``. A copy of each
            column is stored internally.
        functions (list[str] | FunctionSet): Either a list of function names
            (to be resolved via ``FunctionSet.from_names``) or an already
            constructed ``FunctionSet`` instance.
        constant_range (float): Symmetric numeric range used to generate constants.
            Constants are sampled as 100 evenly spaced values in the interval
            ``[-constant_range, constant_range]``.

    Attributes:
        constant_range (float): The symmetric range used to generate constants.
        function_set_ (FunctionSet): The resolved function set used by the GP engine.
        terminals_ (dict[str, Array]): Mapping from terminal names (e.g. ``"x0"``)
            to 1D NumPy arrays of shape ``(n_samples,)``.
        constants_ (dict[str, Array]): Mapping from constant names (e.g. ``"c0"``)
            to 1D NumPy arrays of shape ``(n_samples,)`` with constant values.

    Example:
        >>> X = np.random.randn(5, 2)
        >>> pset = PrimitiveSet(X, functions=["add", "mul"], constant_range=1.0)
        >>> f_add = pset.get_function("add")
        >>> x0 = pset.get_terminal("x0")
        >>> x0.shape
        (5,)
    """

    def __init__(
        self,
        X: np.ndarray,
        functions: Union[list[str], "FunctionSet"],
        constant_range: float,
        X_test: Optional[np.ndarray] = None,
    ) -> None:
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape!r}")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature (n_features >= 1).")

        self.constant_range = float(constant_range)

        if isinstance(functions, list):
            self.function_set_ = FunctionSet.from_names(functions)
        else:
            self.function_set_ = functions

        # Cache terminals for the provided dataset (can already include train+test rows)
        self.terminals_: list[GPTerminal] = [
            GPTerminal(name=f"x{i}", values=X[:, i].copy())
            for i in range(X.shape[1])
        ]
        self._terminal_dict: dict[str, GPTerminal] = {t.name: t for t in self.terminals_}
        self.n_samples = X.shape[0]

        # Cache constants for both train and test modes
        self._build_constants()
        self._validate_invariants()
    
    def _build_constants(self) -> None:
        """Build constant terminals in the configured numeric range.

        Creates 100 evenly spaced constants in the interval
        ``[-constant_range, constant_range]`` and stores them as
        1D arrays with the same number of samples as the input terminals.
        Caches constants for both train and test modes.
        """
        
        constants = np.linspace(-self.constant_range, self.constant_range, num=100)

        self.constants_: list[GPConstant] = []
        self._constant_dict: dict[str, GPConstant] = {}
        for value in constants:
            value = round(value, 4)
            array = np.full(shape=(self.n_samples,), fill_value=value, dtype=float)
            value_str = f"{value:.4f}".rstrip("0").rstrip(".")
            constant = GPConstant(name=f"c{value_str}", values=array)
            self.constants_.append(constant)
            self._constant_dict[constant.name] = constant
                        
    def _validate_invariants(self) -> None:
        """Ensure that the primitive set is in a valid state.
        After this, sampling methods can assume non-empty collections.
        """
        if not getattr(self.function_set_, "functions", None):
            raise ValueError(
                "FunctionSet is empty; PrimitiveSet requires at least one function."
            )

        if not self.terminals_:
            raise ValueError(
                "PrimitiveSet has no terminals; X must have n_features >= 1."
            )

        if not getattr(self, "constants_", None):
            raise ValueError(
                "PrimitiveSet has no constants; check constant_range and _build_constants."
            )


    def get_function(self, name: str) -> GPFunction:
        """Return a GPFunction by name.

        Args:
            name (str): Name of the function to retrieve.

        Returns:
            GPFunction: The function object registered under the given name.

        Raises:
            KeyError: If no function with the given name exists in the function set.
        """
        return self.function_set_.get(name)

    def get_terminal(self, name: str) -> GPTerminal:
        """Return a terminal array (variable or constant) by name.

        Args:
            name (str): Name of the terminal. Expected to be:
                - ``"x{i}"`` for input variables, or

        Returns:
            GPTerminal: The terminal object with its name and values.

        Raises:
            KeyError: If the terminal name does not exist.
        """
        if name in self._terminal_dict:
            return self._terminal_dict[name]
        raise KeyError(f"Unknown terminal name: {name!r}, available: {[t.name for t in self.terminals_]}")
    
    def get_constant(self, name: str) -> Array:
        """Return a constant terminal array by name.

        Args:
            name (str): Name of the constant terminal. Expected to be
                in the format ``"c_{value}"``.

        Returns:
            Array: A 1D NumPy array of shape ``(n_samples,)`` representing the constant.

        Raises:
            KeyError: If the constant name does not exist.
        """
        if name in self._constant_dict:
            return self._constant_dict[name].values
        # Fallback to list iteration for backward compatibility
        for constant in self.constants_:
            if constant.name == name:
                return constant.values
        raise KeyError(f"Unknown constant name: {name!r}, available: {[c.name for c in self.constants_]}")

    def sample_function(
        self,
        rng: np.random.Generator,
    ) -> GPFunction:
        """Sample a random function from the function set.

        Args:
            rng (np.random.Generator, optional): Random number generator to use.
                If None, a new default generator is created.

        Returns:
            GPFunction: A randomly selected function.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> f = pset.sample_function(rng)
            >>> isinstance(f.name, str)
            True
        """
        functions = self.function_set_.functions
        idx = rng.integers(0, len(functions))
        return functions[idx]

    def sample_constant(
        self,
        rng: np.random.Generator,
    ) -> GPConstant:
        """Sample a random constant terminal.

        Args:
            rng (np.random.Generator, optional): Random number generator to use.
                If None, a new default generator is created.

        Returns:
            GPConstant: A randomly selected constant with its name and values.

        Raises:
            RuntimeError: If no constants have been built.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> name, values = pset.sample_constant(rng)
            >>> name.startswith("c")
            True
        """
        idx = rng.integers(0, len(self.constants_))
        constant = self.constants_[idx]
        return constant

    def sample_terminal(
        self,
        rng: np.random.Generator,
    ) -> GPTerminal:
        """Sample a random terminal (variable or optionally constant).

        Args:
            rng (np.random.Generator, optional): Random number generator to use.
                If None, a new default generator is created.

        Returns:
            GPTerminal: A randomly selected terminal with its name and values.

        Raises:
            RuntimeError: If no terminals are available to sample from.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> name, values = pset.sample_terminal(rng)
            >>> name.startswith("x")
            True
        """
        idx = rng.integers(0, len(self.terminals_))
        terminal = self.terminals_[idx]
        return terminal
    
    # ======================================================================
    # Specialist Support for PSSR (Piecewise Specialist Symbolic Regression)
    # ======================================================================
    
    def set_specialists(
        self,
        specialists: dict[str, "Specialist"],
    ) -> None:
        """Set the specialists for ensemble tree evaluation.
        
        Specialists are pre-trained GP individuals that act as terminal
        nodes in ensemble trees. Their outputs are pre-computed and cached
        for fast ensemble evaluation.
        
        Semantics are stored as combined arrays (train + test) with slices
        for efficient access, similar to how GPRegressor handles it.
        
        The train/test slices are automatically extracted from the specialists'
        underlying Individual objects, which should all have the same slices
        since they're evaluated on the same combined data.
        
        Args:
            specialists: Dictionary mapping specialist names (e.g., "S_0") 
                to Specialist objects.
                
        Example:
            >>> from pssr.pssr_model.specialist import create_specialists_from_population
            >>> specialists = create_specialists_from_population(population, X_combined)
            >>> pset.set_specialists(specialists)
        """
        if not specialists:
            raise ValueError("specialists dictionary cannot be empty")
        
        self._specialists: dict[str, "Specialist"] = specialists
        self._specialist_names: list[str] = list(specialists.keys())
        
        # Extract slices from the first specialist's individual
        # All specialists should have the same slices since they use the same data
        first_specialist = next(iter(specialists.values()))
        first_individual = first_specialist.individual
        
        # Get slices from the individual (they should be set via set_slices())
        self._specialist_train_slice = getattr(first_individual, '_train_slice', None)
        self._specialist_test_slice = getattr(first_individual, '_test_slice', None)
        
        # Cache combined specialist semantics for fast O(1) lookup
        # These are numpy arrays directly, avoiding method call overhead
        self._specialist_semantics: dict[str, Array] = {}
        
        for name, specialist in specialists.items():
            if specialist.semantics is not None:
                self._specialist_semantics[name] = specialist.semantics
    
    def cache_specialist_semantics(self, X: np.ndarray) -> None:
        """Compute and cache combined semantics for all specialists.
        
        This should be called after set_specialists() if specialists
        don't already have their semantics computed, or if the data changed.
        
        Args:
            X: Combined data (train + test) to compute semantics
        """
        if not hasattr(self, '_specialists'):
            raise AttributeError("Specialists not set. Call set_specialists() first.")
        
        self._specialist_semantics = {}
        
        for name, specialist in self._specialists.items():
            specialist.compute_semantics(X)
            if specialist.semantics is not None:
                self._specialist_semantics[name] = specialist.semantics
    
    @property
    def specialists(self) -> dict[str, "Specialist"]:
        """Get the dictionary of specialists.
        
        Returns:
            Dictionary mapping specialist names to Specialist objects.
            
        Raises:
            AttributeError: If specialists haven't been set.
        """
        if not hasattr(self, '_specialists'):
            raise AttributeError(
                "Specialists not set. Call set_specialists() first."
            )
        return self._specialists
    
    @property
    def specialist_names(self) -> list[str]:
        """Get the list of specialist names.
        
        Returns:
            List of specialist name strings.
            
        Raises:
            AttributeError: If specialists haven't been set.
        """
        if not hasattr(self, '_specialist_names'):
            raise AttributeError(
                "Specialists not set. Call set_specialists() first."
            )
        return self._specialist_names
    
    def has_specialists(self) -> bool:
        """Check if specialists have been set.
        
        Returns:
            True if specialists are available, False otherwise.
        """
        return hasattr(self, '_specialists') and len(self._specialists) > 0
    
    def get_specialist(self, name: str) -> "Specialist":
        """Get a specialist by name.
        
        Args:
            name: The specialist name (e.g., "S_0").
            
        Returns:
            The Specialist object.
            
        Raises:
            KeyError: If the specialist name doesn't exist.
            AttributeError: If specialists haven't been set.
        """
        if not hasattr(self, '_specialists'):
            raise AttributeError(
                "Specialists not set. Call set_specialists() first."
            )
        if name not in self._specialists:
            raise KeyError(
                f"Unknown specialist name: {name!r}, "
                f"available: {self._specialist_names}"
            )
        return self._specialists[name]
    
    def get_specialist_semantics(
        self,
        name: str,
        testing: bool = False,
        predict: bool = False,
        X: Optional[np.ndarray] = None,
    ) -> Array:
        """Get cached semantics for a specialist.
        
        Uses O(1) dict lookup for fast retrieval of cached semantics.
        Combined semantics are sliced to get train/test portions.
        Falls back to computing predictions only when predict=True.
        
        Args:
            name: The specialist name.
            testing: If True, return test semantics; else return train semantics.
            predict: If True, compute fresh predictions on X.
            X: Input data (required if predict=True).
            
        Returns:
            The semantic (output) values as a 1D array.
            
        Raises:
            KeyError: If the specialist name doesn't exist.
            ValueError: If semantics not available and predict=False.
        """
        # For prediction on new data, must compute fresh
        if predict:
            if X is None:
                raise ValueError("X must be provided when predict=True")
            specialist = self.get_specialist(name)
            return specialist.predict(X)
        
        # Fast O(1) lookup from cached combined semantics
        if not hasattr(self, '_specialist_semantics') or name not in self._specialist_semantics:
            raise ValueError(
                f"Semantics not cached for specialist {name}. "
                "Call set_specialists() or cache_specialist_semantics() first."
            )
        
        combined_semantics = self._specialist_semantics[name]
        
        # Slice to get train or test portion
        if testing:
            if self._specialist_test_slice is None:
                raise ValueError("Test slice not set. No test data available.")
            return combined_semantics[self._specialist_test_slice]
        else:
            if self._specialist_train_slice is None:
                # No slice set, return full array (assume all train)
                return combined_semantics
            return combined_semantics[self._specialist_train_slice]
    
    def sample_specialist(self, rng: np.random.Generator) -> "Specialist":
        """Sample a random specialist.
        
        Args:
            rng: Random number generator.
            
        Returns:
            A randomly selected Specialist.
            
        Raises:
            AttributeError: If specialists haven't been set.
            ValueError: If no specialists are available.
        """
        if not hasattr(self, '_specialists') or not self._specialists:
            raise ValueError("No specialists available. Call set_specialists() first.")
        
        idx = rng.integers(0, len(self._specialist_names))
        name = self._specialist_names[idx]
        return self._specialists[name]
    
    def sample_specialist_name(self, rng: np.random.Generator) -> str:
        """Sample a random specialist name.
        
        Args:
            rng: Random number generator.
            
        Returns:
            A randomly selected specialist name.
            
        Raises:
            AttributeError: If specialists haven't been set.
            ValueError: If no specialists are available.
        """
        if not hasattr(self, '_specialist_names') or not self._specialist_names:
            raise ValueError("No specialists available. Call set_specialists() first.")
        
        idx = rng.integers(0, len(self._specialist_names))
        return self._specialist_names[idx]
    
    def n_specialists(self) -> int:
        """Get the number of specialists.
        
        Returns:
            Number of specialists, or 0 if not set.
        """
        if not hasattr(self, '_specialists'):
            return 0
        return len(self._specialists)