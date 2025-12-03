from typing import Optional, Union, Callable

import numpy as np
import numpy.typing as npt

from pssr.core.primitives import PrimitiveSet
Array = npt.NDArray[np.float64]
TreeRepr = Union[tuple, str]  # Tree is either (func_name, *children) or terminal name


class Individual:
    """
    Represents a Genetic Programming individual (candidate solution).
    
    An Individual contains:
    - A tree representation (genotype)
    - Fitness metrics (RMSE, case-wise errors)
    - Size metrics (depth, total_nodes)
    - Methods for evaluation and manipulation
    
    Attributes
    ----------
    tree : TreeRepr
        Tree representation: tuple for function nodes, str for terminals
    fitness : Optional[float]
        Overall fitness (RMSE) on training data
    errors_case : Optional[Array]
        Case-wise errors, shape (n_cases,)
    depth : int
        Maximum depth of the tree
    total_nodes : int
        Total number of nodes in the tree
    """
    
    def __init__(
        self,
        tree: TreeRepr,
        primitive_set: Optional[PrimitiveSet] = None,
        depth: Optional[int] = None,
        total_nodes: Optional[int] = None,
    ):
        """
        Initialize an Individual.
        
        Parameters
        ----------
        tree : TreeRepr
            Tree representation: tuple (func_name, *children) or str (terminal name)
        primitive_set : PrimitiveSet, optional
            Primitive set for evaluation. If None, depth/nodes must be provided.
        depth : int, optional
            Pre-computed depth. If None, will be calculated.
        total_nodes : int, optional
            Pre-computed node count. If None, will be calculated.
        """
        self.tree = tree
        self.primitive_set = primitive_set
        
        if depth is not None and total_nodes is not None:
            self.depth = depth
            self.total_nodes = total_nodes
        else:
            self.depth, self.total_nodes = self._calculate_depth_and_nodes(tree)
        
        self.fitness: Optional[float] = None
        self.test_fitness: Optional[float] = None
        self.errors_case: Optional[Array] = None
        self.size: int = self.total_nodes  # Alias for compatibility
        
        self.train_semantics: Optional[Array] = None
        self.test_semantics: Optional[Array] = None
        
        # Slice indices for combined evaluation
        self._train_slice: Optional[slice] = None
        self._test_slice: Optional[slice] = None
    
    def set_slices(self, train_slice: slice, test_slice: Optional[slice] = None) -> None:
        """Set the slice indices for train/test portions of combined data."""
        self._train_slice = train_slice
        self._test_slice = test_slice
    
    def _calculate_depth_and_nodes(self, tree: TreeRepr) -> tuple[int, int]:
        """
        Calculate depth and node count of the tree.
        
        Parameters
        ----------
        tree : TreeRepr
            Tree representation
            
        Returns
        -------
        tuple[int, int]
            (depth, total_nodes)
        """
        if isinstance(tree, tuple):
            # Function node: (func_name, child1, child2, ...)
            _ = tree[0]
            children = tree[1:]
            
            if not children:
                return 1, 1
            
            child_depths = []
            total = 1  # Count this node
            for child in children:
                child_depth, child_nodes = self._calculate_depth_and_nodes(child)
                child_depths.append(child_depth)
                total += child_nodes
            
            depth = 1 + max(child_depths) if child_depths else 1
            return depth, total
        else:
            # Terminal node
            return 1, 1
    
    def calculate_semantics(self, X: Array,
                      primitive_set: Optional[PrimitiveSet] = None) -> Array:
        """
        Calculate the semantics of the individual on the full dataset (train + test combined).
        
        If slices are set via set_slices(), automatically assigns train_semantics
        and test_semantics from the combined result.
        
        Parameters
        ----------
        X : Array
            Combined input data (train + test)
        primitive_set : Optional[PrimitiveSet]
            Primitive set for evaluation
            
        Returns
        -------
        Array
            Combined semantics
        """
        if primitive_set is None:
            if self.primitive_set is None:
                raise ValueError("primitive_set must be provided either as argument or in __init__")
            primitive_set = self.primitive_set
        
        full_semantics = self._execute_tree(self.tree, X, primitive_set)
        
        # Auto-assign train/test semantics if slices are set -> pointer assignement 
        if self._train_slice is not None:
            self.train_semantics = full_semantics[self._train_slice]
        if self._test_slice is not None:
            self.test_semantics = full_semantics[self._test_slice]
        
        return full_semantics
    
    def evaluate(
        self,
        y: Array,
        y_test: Optional[Array] = None,
        fitness_function: Optional[Callable[[Array, Array], float]] = None,
    ) -> float:
        """
        Calculate fitness (default RMSE) of the individual.
        
        Uses pre-computed train_semantics (and optionally test_semantics) to calculate
        fitness. Semantics must be calculated first via calculate_semantics().
        
        Parameters
        ----------
        y : Array
            Training target values, shape (n_train_samples,)
        y_test : Optional[Array]
            Test target values, shape (n_test_samples,). If provided and test_semantics
            is available, also calculates test_fitness.
        fitness_function : Optional[Callable[[Array, Array], float]]
            Custom fitness function. Must accept (y_true, y_pred) and return a float.
            Default is RMSE: sqrt(mean((y_true - y_pred)^2))
            
        Returns
        -------
        float
            Training fitness value (RMSE by default)
        """
        if self.train_semantics is None:
            raise ValueError(
                "train_semantics not calculated. Call calculate_semantics() or evaluate_full() first."
            )
        
        # Default fitness function: RMSE
        if fitness_function is None:
            def rmse(y_true: Array, y_pred: Array) -> float:
                errors = y_true - y_pred
                mse = np.mean(errors**2)
                return float(np.sqrt(mse))
            fitness_function = rmse
        
        # Calculate training errors and fitness
        self.fitness = fitness_function(y, self.train_semantics)
        
        # Calculate test fitness if test data and semantics are available
        if y_test is not None and self.test_semantics is not None:
            self.test_fitness = fitness_function(y_test, self.test_semantics)
        
        return self.fitness
                
    def _execute_tree(
        self,
        tree: TreeRepr,
        X: Array,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """
        Recursively execute the tree representation.
        
        Parameters
        ----------
        tree : TreeRepr
            Tree representation to evaluate
        X : Array
            Input data of shape (n_samples, n_features)
        primitive_set : PrimitiveSet
            Primitive set containing functions, terminals, and constants
            
        Returns
        -------
        Array
            Evaluated output of shape (n_samples,)
        """
        if isinstance(tree, tuple):
            # Function node: (func_name, child1, child2, ...)
            func_name = tree[0]
            children = tree[1:]
            
            # Get function from primitive set (direct access for performance)
            gp_func = primitive_set.function_set_.get(func_name)
            
            # Recursively evaluate children
            child_results = [
                self._execute_tree(child, X, primitive_set)
                for child in children
            ]
            
            # Apply function to children
            result = gp_func.func(*child_results)
            
            # Bound values to prevent overflow (in-place operation for performance)
            result.clip(-1e12, 1e12, out=result)
            return result
        
        else:
            # Terminal node: string name
            terminal_name = tree
            
            # Check if it's a variable terminal (x0, x1, ...) - O(1) dict lookup
            if terminal_name in primitive_set._terminal_dict:
                return primitive_set._terminal_dict[terminal_name].values
            
            # Check if it's a constant (c0.5, c-1.0, ...) - O(1) dict lookup
            if terminal_name in primitive_set._constant_dict:
                return primitive_set._constant_dict[terminal_name].values
            
            raise ValueError(f"Unknown terminal: {terminal_name!r}")
    
    def predict(
        self,
        X: Array,
        tree: Optional[TreeRepr] = None,
        primitive_set: Optional[PrimitiveSet] = None,
    ) -> Array:
        """
        Recursively execute the tree representation for prediction.
        
        This method is used during prediction/testing and extracts values directly
        from X instead of using stored primitive set values. This allows prediction
        on different datasets (e.g., test set) without modifying the primitive set.
        
        Parameters
        ----------
        X : Array
            Input data of shape (n_samples, n_features)
        tree : TreeRepr
            Tree representation to evaluate
        primitive_set : PrimitiveSet
            Primitive set containing functions (only function_set_ is used)
            
        Returns
        -------
        Array
            Evaluated output of shape (n_samples,)
        """
        if primitive_set is None:
            if self.primitive_set is None:
                raise ValueError("primitive_set must be provided either as argument or in __init__")
            primitive_set = self.primitive_set
        
        if tree is None:
            tree = self.tree
        
        if isinstance(tree, tuple):
            # Function node: (func_name, child1, child2, ...)
            func_name = tree[0]
            children = tree[1:]
            
            # Get function from primitive set
            gp_func = primitive_set.function_set_.get(func_name)
            
            # Recursively evaluate children
            child_results = [
                self.predict(
                    X=X,
                    tree=child,
                    primitive_set=primitive_set,
                )
                for child in children
            ]
            
            # Apply function to children
            result = gp_func.func(*child_results)
            
            # Bound values to prevent overflow
            result.clip(-1e12, 1e12, out=result)
            return result
        
        else:
            # Terminal node: string name
            terminal_name = tree
            
            # Check if it's a variable terminal (x0, x1, ...)
            if terminal_name.startswith('x') and terminal_name[1:].isdigit():
                # Extract column index and use X directly
                column_idx = int(terminal_name[1:])
                if column_idx >= X.shape[1]:
                    raise ValueError(
                        f"Terminal {terminal_name!r} references column {column_idx}, "
                        f"but X only has {X.shape[1]} columns"
                    )
                return X[:, column_idx].copy()
            
            # Check if it's a constant (c0.5, c-1.0, ...)
            if terminal_name.startswith('c'):
                # Parse constant value from name (e.g., "c0.5" -> 0.5, "c-1.0" -> -1.0)
                try:
                    constant_value = float(terminal_name[1:])
                    # Broadcast to match X's number of samples
                    return np.full(X.shape[0], constant_value, dtype=float)
                except ValueError:
                    raise ValueError(f"Invalid constant terminal format: {terminal_name!r}")
            
            raise ValueError(f"Unknown terminal: {terminal_name!r}")
    
    def copy(self) -> "Individual":
        """
        Create a deep copy of this individual.
        
        Returns
        -------
        Individual
            A new Individual with copied tree and metrics
        """
        # Deep copy the tree structure
        tree_copy = self._deep_copy_tree(self.tree)
        
        new_ind = Individual(
            tree=tree_copy,
            primitive_set=self.primitive_set,
            depth=self.depth,
            total_nodes=self.total_nodes,
        )
        
        # Copy fitness metrics if they exist
        if self.fitness is not None:
            new_ind.fitness = self.fitness
        if self.errors_case is not None:
            new_ind.errors_case = self.errors_case.copy()
        
        # Copy slice indices
        new_ind._train_slice = self._train_slice
        new_ind._test_slice = self._test_slice
        
        return new_ind
    
    def _deep_copy_tree(self, tree: TreeRepr) -> TreeRepr:
        """Recursively deep copy tree structure."""
        if isinstance(tree, tuple):
            return tuple(self._deep_copy_tree(child) for child in tree)
        else:
            return tree  # Strings are immutable
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        fitness_str = f", fitness={self.fitness:.6f}" if self.fitness is not None else ""
        return f"Individual(depth={self.depth}, nodes={self.total_nodes}{fitness_str})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self._tree_to_string(self.tree)
    
    def _tree_to_string(self, tree: TreeRepr, indent: int = 0) -> str:
        """Convert tree to readable string format."""
        if isinstance(tree, tuple):
            func_name = tree[0]
            children = tree[1:]
            if not children:
                return func_name
            child_strs = [self._tree_to_string(child, indent + 1) for child in children]
            return f"{func_name}({', '.join(child_strs)})"
        else:
            return str(tree)