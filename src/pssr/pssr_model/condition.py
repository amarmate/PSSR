"""
Condition class for PSSR ensemble trees.

A Condition is a GP tree that is used as the predicate in conditional 
ensemble nodes. The condition evaluates to a numeric value, and if > 0,
the true branch is taken; otherwise the false branch.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from pssr.core.primitives import PrimitiveSet

Array = npt.NDArray[np.float64]
TreeRepr = Union[tuple, str]  # Tree is either (func_name, *children) or terminal name


class Condition:
    """
    Represents a condition tree for ensemble branching decisions.
    
    A Condition is a standard GP tree that outputs numeric values.
    When evaluated, points where output > 0 follow the true branch,
    and points where output <= 0 follow the false branch.
    
    Semantics are stored as a single combined array (train + test) and
    sliced as needed, similar to how specialists are handled.
    
    Attributes
    ----------
    tree : TreeRepr
        Tree representation: tuple for function nodes, str for terminals
    depth : int
        Maximum depth of the condition tree
    nodes_count : int
        Total number of nodes in the condition tree
    semantics : Optional[Array]
        Cached condition outputs on combined data (train + test)
    """
    
    def __init__(
        self,
        tree: TreeRepr,
        depth: Optional[int] = None,
        nodes_count: Optional[int] = None,
    ):
        """
        Initialize a Condition.
        
        Parameters
        ----------
        tree : TreeRepr
            Tree representation (tuple or string)
        depth : Optional[int]
            Pre-computed depth. If None, will be calculated.
        nodes_count : Optional[int]
            Pre-computed node count. If None, will be calculated.
        """
        self.tree = tree
        
        if depth is not None and nodes_count is not None:
            self.depth = depth
            self.nodes_count = nodes_count
        else:
            self.depth, self.nodes_count = self._calculate_depth_and_nodes(tree)
        
        # Combined semantics (train + test in one array)
        self.semantics: Optional[Array] = None
        
        # Slice indices for train/test portions
        self._train_slice: Optional[slice] = None
        self._test_slice: Optional[slice] = None
    
    def set_slices(self, train_slice: slice, test_slice: Optional[slice] = None) -> None:
        """Set the slice indices for train/test portions of combined data."""
        self._train_slice = train_slice
        self._test_slice = test_slice
    
    @property
    def train_semantics(self) -> Optional[Array]:
        """Get training portion of semantics."""
        if self.semantics is None:
            return None
        if self._train_slice is not None:
            return self.semantics[self._train_slice]
        return self.semantics
    
    @property
    def test_semantics(self) -> Optional[Array]:
        """Get test portion of semantics."""
        if self.semantics is None or self._test_slice is None:
            return None
        return self.semantics[self._test_slice]
    
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
    
    def apply_tree(
        self,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """
        Evaluate the condition tree using cached primitive set values (fast).
        
        Parameters
        ----------
        primitive_set : PrimitiveSet
            Primitive set for evaluation (uses cached terminal/constant values)
            
        Returns
        -------
        Array
            Condition outputs, shape (n_samples,)
        """
        return self._execute_tree(self.tree, primitive_set)
    
    def _execute_tree(
        self,
        tree: TreeRepr,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """
        Recursively execute the tree representation (fast, uses cached values).
        
        Parameters
        ----------
        tree : TreeRepr
            Tree representation to evaluate
        primitive_set : PrimitiveSet
            Primitive set containing functions, terminals, and constants
            with cached values
            
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
                self._execute_tree(child, primitive_set)
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
    
    def calculate_semantics(
        self,
        primitive_set: PrimitiveSet,
    ) -> None:
        """
        Calculate and cache combined semantics for the condition.
        
        Uses the primitive set's cached terminal values for fast evaluation.
        
        Parameters
        ----------
        primitive_set : PrimitiveSet
            Primitive set for evaluation (with combined X already loaded)
        """
        # The primitive set already has combined X data loaded
        # We evaluate using _execute_tree which uses cached values from primitive_set
        self.semantics = self._execute_tree(self.tree, primitive_set)
    
    def get_semantics(
        self,
        train_slice: Optional[slice] = None,
        test_slice: Optional[slice] = None,
        testing: bool = False,
    ) -> Array:
        """
        Get cached semantics, optionally sliced.
        
        Parameters
        ----------
        train_slice : Optional[slice]
            Slice for training portion
        test_slice : Optional[slice]
            Slice for test portion
        testing : bool
            If True, return test portion; else return train portion
            
        Returns
        -------
        Array
            Cached semantics (possibly sliced)
            
        Raises
        ------
        ValueError
            If semantics haven't been calculated
        """
        if self.semantics is None:
            raise ValueError("Semantics not calculated. Call calculate_semantics first.")
        
        if testing:
            if test_slice is not None:
                return self.semantics[test_slice]
            else:
                raise ValueError("Test slice not provided but testing=True")
        else:
            if train_slice is not None:
                return self.semantics[train_slice]
            else:
                # No slice, return full array
                return self.semantics
    
    def predict(
        self,
        X: Array,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """
        Compute condition outputs on new data.
        
        This method extracts values directly from X rather than using cached
        primitive set values, making it suitable for prediction on new data.
        
        Parameters
        ----------
        X : Array
            Input data of shape (n_samples, n_features)
        primitive_set : PrimitiveSet
            Primitive set containing functions (only function_set_ is used)
            
        Returns
        -------
        Array
            Condition outputs of shape (n_samples,)
        """
        return self._predict_tree(self.tree, X, primitive_set)
    
    def _predict_tree(
        self,
        tree: TreeRepr,
        X: Array,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """
        Recursively execute tree for prediction on new data.
        
        Unlike _execute_tree which uses cached primitive set values,
        this method extracts values directly from X.
        
        Parameters
        ----------
        tree : TreeRepr
            Tree representation to evaluate
        X : Array
            Input data of shape (n_samples, n_features)
        primitive_set : PrimitiveSet
            Primitive set containing functions
            
        Returns
        -------
        Array
            Evaluated output of shape (n_samples,)
        """
        if isinstance(tree, tuple):
            # Function node: (func_name, child1, child2, ...)
            func_name = tree[0]
            children = tree[1:]
            
            # Get function from primitive set
            gp_func = primitive_set.function_set_.get(func_name)
            
            # Recursively evaluate children
            child_results = [
                self._predict_tree(child, X, primitive_set)
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
    
    def copy(self) -> "Condition":
        """
        Create a deep copy of this condition.
        
        Returns
        -------
        Condition
            A new Condition with copied tree
        """
        tree_copy = self._deep_copy_tree(self.tree)
        new_cond = Condition(
            tree=tree_copy,
            depth=self.depth,
            nodes_count=self.nodes_count,
        )
        
        if self.semantics is not None:
            new_cond.semantics = self.semantics.copy()
        
        # Copy slice indices
        new_cond._train_slice = self._train_slice
        new_cond._test_slice = self._test_slice
        
        return new_cond
    
    def _deep_copy_tree(self, tree: TreeRepr) -> TreeRepr:
        """Recursively deep copy tree structure."""
        if isinstance(tree, tuple):
            return tuple(self._deep_copy_tree(child) for child in tree)
        else:
            return tree
    
    def __repr__(self) -> str:
        return f"Condition(depth={self.depth}, nodes={self.nodes_count})"
    
    def __str__(self) -> str:
        return self._tree_to_string(self.tree)
    
    def _tree_to_string(self, tree: TreeRepr) -> str:
        """Convert tree to readable string format."""
        if isinstance(tree, tuple):
            func_name = tree[0]
            children = tree[1:]
            if not children:
                return func_name
            child_strs = [self._tree_to_string(child) for child in children]
            return f"{func_name}({', '.join(child_strs)})"
        else:
            return str(tree)
    
    def get_tree_representation(self, indent: str = "") -> str:
        """
        Get a formatted tree representation with indentation.
        
        Parameters
        ----------
        indent : str
            Current indentation string
            
        Returns
        -------
        str
            Formatted tree string
        """
        return self._format_tree(self.tree, indent)
    
    def _format_tree(self, tree: TreeRepr, indent: str) -> str:
        """Format tree with indentation."""
        if isinstance(tree, tuple):
            func_name = tree[0]
            children = tree[1:]
            
            if not children:
                return f"{indent}{func_name}\n"
            
            result = f"{indent}{func_name}(\n"
            for child in children:
                result += self._format_tree(child, indent + "  ")
            result += f"{indent})\n"
            return result
        else:
            return f"{indent}{tree}\n"
