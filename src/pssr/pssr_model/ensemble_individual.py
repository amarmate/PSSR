"""
Ensemble Individual for PSSR (Piecewise Specialist Symbolic Regression).

An EnsembleIndividual represents a conditional tree that routes inputs
to different specialists based on learned conditions.

Tree structure:
- (Condition, true_branch, false_branch) - conditional node
- "S_i" - specialist terminal (leaf)

Evaluation:
- For each sample, evaluate condition
- If condition > 0: use true_branch output
- Else: use false_branch output
"""

from typing import Optional, Union, Callable

import numpy as np
import numpy.typing as npt

from pssr.core.primitives import PrimitiveSet
from pssr.pssr_model.condition import Condition

Array = npt.NDArray[np.float64]

# Ensemble tree can be:
# - A tuple (Condition, true_branch, false_branch) for conditional nodes
# - A string "S_i" for specialist terminals
EnsembleTreeRepr = Union[tuple, str]


class EnsembleIndividual:
    """
    Represents a PSSR ensemble individual (conditional tree with specialists).
    
    An EnsembleIndividual is a tree structure where:
    - Internal nodes are conditional: (Condition, true_branch, false_branch)
    - Leaf nodes are specialists: "S_0", "S_1", etc.
    
    The tree evaluates inputs by routing them through conditions to specialists,
    then combining specialist outputs based on the routing decisions.
    
    Follows the same interface as Individual for Population compatibility.
    
    Attributes
    ----------
    collection : EnsembleTreeRepr
        The tree representation
    depth : int
        Maximum depth of the ensemble tree (not counting specialist internals)
    total_nodes : int
        Total nodes including nodes inside specialists
    nodes_count : int
        Number of nodes in the ensemble tree (conditions + specialist references)
    fitness : Optional[float]
        Training fitness (RMSE)
    test_fitness : Optional[float]
        Test fitness (RMSE)
    train_semantics : Optional[Array]
        Cached predictions on training data
    test_semantics : Optional[Array]
        Cached predictions on test data
    """
    
    def __init__(
        self,
        collection: EnsembleTreeRepr,
        primitive_set: Optional[PrimitiveSet] = None,
        depth: Optional[int] = None,
        nodes_count: Optional[int] = None,
        total_nodes: Optional[int] = None,
    ):
        """
        Initialize an EnsembleIndividual.
        
        Parameters
        ----------
        collection : EnsembleTreeRepr
            Tree representation
        primitive_set : Optional[PrimitiveSet]
            Primitive set with specialists for evaluation
        depth : Optional[int]
            Pre-computed depth
        nodes_count : Optional[int]
            Pre-computed node count (ensemble nodes only)
        total_nodes : Optional[int]
            Pre-computed total nodes (including specialist internals)
        """
        self.collection = collection
        self.primitive_set = primitive_set
        
        if depth is not None and nodes_count is not None and total_nodes is not None:
            self.depth = depth
            self.nodes_count = nodes_count
            self.total_nodes = total_nodes
        else:
            self._compute_metrics()
        
        # Alias for compatibility
        self.size: int = self.total_nodes
        
        # Fitness metrics (same as Individual)
        self.fitness: Optional[float] = None
        self.test_fitness: Optional[float] = None
        self.errors_case: Optional[Array] = None
        
        # Semantic arrays (same as Individual)
        self.train_semantics: Optional[Array] = None
        self.test_semantics: Optional[Array] = None
        
        # Slice indices for combined evaluation (same as Individual)
        self._train_slice: Optional[slice] = None
        self._test_slice: Optional[slice] = None
    
    def set_slices(self, train_slice: slice, test_slice: Optional[slice] = None) -> None:
        """Set the slice indices for train/test portions of combined data."""
        self._train_slice = train_slice
        self._test_slice = test_slice
    
    def _compute_metrics(self) -> None:
        """Compute depth, nodes_count, and total_nodes from the tree structure."""
        if self.primitive_set is not None and self.primitive_set.has_specialists():
            self.depth, self.nodes_count, self.total_nodes = self._tree_depth_and_nodes(
                self.collection, self.primitive_set
            )
        else:
            # Without specialists, we can only count ensemble structure
            self.depth, self.nodes_count = self._basic_depth_and_nodes(self.collection)
            self.total_nodes = self.nodes_count
    
    def _tree_depth_and_nodes(
        self,
        tree: EnsembleTreeRepr,
        primitive_set: PrimitiveSet,
        depth: int = 1,
    ) -> tuple[int, int, int]:
        """Recursively compute depth, nodes, and total_nodes."""
        if isinstance(tree, tuple):
            condition = tree[0]
            true_branch = tree[1]
            false_branch = tree[2]
            
            cond_nodes = condition.nodes_count if isinstance(condition, Condition) else 1
            
            true_depth, true_nodes, true_total = self._tree_depth_and_nodes(
                true_branch, primitive_set, depth + 1
            )
            false_depth, false_nodes, false_total = self._tree_depth_and_nodes(
                false_branch, primitive_set, depth + 1
            )
            
            max_depth = max(true_depth, false_depth)
            nodes = 1 + true_nodes + false_nodes
            total = cond_nodes + true_total + false_total
            
            return max_depth, nodes, total
        
        elif isinstance(tree, str):
            try:
                specialist = primitive_set.get_specialist(tree)
                return depth, 1, specialist.nodes_count
            except (KeyError, AttributeError):
                return depth, 1, 1
        
        else:
            if isinstance(tree, Condition):
                return depth, 1, tree.nodes_count
            return depth, 1, 1
    
    def _basic_depth_and_nodes(self, tree: EnsembleTreeRepr) -> tuple[int, int]:
        """Compute depth and nodes without specialist info."""
        if isinstance(tree, tuple):
            children = tree[1:]
            if not children:
                return 1, 1
            
            child_stats = [self._basic_depth_and_nodes(child) for child in children]
            max_depth = 1 + max(d for d, _ in child_stats)
            total_nodes = 1 + sum(n for _, n in child_stats)
            return max_depth, total_nodes
        else:
            return 1, 1
    
    # ======================================================================
    # Evaluation Methods (Population Compatible)
    # ======================================================================
    
    def calculate_semantics(
        self,
        X: Optional[Array] = None,
        primitive_set: Optional[PrimitiveSet] = None,
    ) -> Array:
        """
        Calculate the semantics of the individual on the full dataset (train + test combined).
        
        If slices are set via set_slices(), automatically assigns train_semantics
        and test_semantics from the combined result.
        
        For EnsembleIndividual, this uses cached primitive set values.
        X is ignored since we use pre-cached values in the primitive set.
        
        Parameters
        ----------
        X : Optional[Array]
            Combined input data (train + test). Ignored - kept for interface compatibility.
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
        
        # Calculate condition semantics first
        self._calculate_condition_semantics(self.collection, primitive_set)
        
        # Execute tree using cached semantics
        full_semantics = self._execute_tree(self.collection, primitive_set)
        
        # Auto-assign train/test semantics if slices are set
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
                "train_semantics not calculated. Call calculate_semantics() first."
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
    
    def _calculate_condition_semantics(
        self,
        tree: EnsembleTreeRepr,
        primitive_set: PrimitiveSet,
    ) -> None:
        """Recursively calculate and cache condition semantics."""
        if isinstance(tree, tuple):
            condition = tree[0]
            if isinstance(condition, Condition):
                condition.calculate_semantics(primitive_set)
            
            # Recurse into branches
            self._calculate_condition_semantics(tree[1], primitive_set)
            self._calculate_condition_semantics(tree[2], primitive_set)
    
    def _execute_tree(
        self,
        tree: EnsembleTreeRepr,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """
        Recursively execute the ensemble tree using cached semantics.
        
        All condition and specialist semantics should already be calculated.
        """
        if isinstance(tree, tuple):
            condition = tree[0]
            true_branch = tree[1]
            false_branch = tree[2]
            
            # Get cached condition semantics
            cond_semantics = condition.semantics
            if cond_semantics is None:
                raise ValueError("Condition semantics not calculated")
            
            # Create mask: condition > 0 -> true branch
            mask = cond_semantics > 0
            
            # Evaluate branches
            true_output = self._execute_tree(true_branch, primitive_set)
            false_output = self._execute_tree(false_branch, primitive_set)
            
            # Combine using mask
            return np.where(mask, true_output, false_output)
        
        elif isinstance(tree, str):
            # Specialist terminal - get combined semantics
            semantics = primitive_set._specialist_semantics.get(tree)
            if semantics is None:
                raise ValueError(f"Specialist semantics not cached for {tree}")
            return semantics
        
        else:
            raise ValueError(f"Unknown tree node type: {type(tree)}")
    
    # ======================================================================
    # Prediction (for new data)
    # ======================================================================
    
    def predict(
        self,
        X: Array,
        tree: Optional[EnsembleTreeRepr] = None,
        primitive_set: Optional[PrimitiveSet] = None,
    ) -> Array:
        """
        Make predictions on new data.
        
        This method computes fresh predictions by extracting values directly
        from X, similar to Individual.predict().
        
        Parameters
        ----------
        X : Array
            Input data
        tree : Optional[EnsembleTreeRepr]
            Tree to evaluate. Uses self.collection if None.
        primitive_set : Optional[PrimitiveSet]
            Primitive set with specialists
            
        Returns
        -------
        Array
            Predictions
        """
        if primitive_set is None:
            primitive_set = self.primitive_set
        if primitive_set is None:
            raise ValueError("primitive_set must be provided or set on individual")
        
        if tree is None:
            tree = self.collection
        
        return self._predict_tree(tree, X, primitive_set)
    
    def _predict_tree(
        self,
        tree: EnsembleTreeRepr,
        X: Array,
        primitive_set: PrimitiveSet,
    ) -> Array:
        """Recursively execute tree for prediction on new data."""
        if isinstance(tree, tuple):
            condition = tree[0]
            true_branch = tree[1]
            false_branch = tree[2]
            
            # Compute condition on new data
            cond_output = condition.predict(X, primitive_set)
            
            # Create mask
            mask = cond_output > 0
            
            # Evaluate branches
            true_output = self._predict_tree(true_branch, X, primitive_set)
            false_output = self._predict_tree(false_branch, X, primitive_set)
            
            return np.where(mask, true_output, false_output)
        
        elif isinstance(tree, str):
            # Specialist terminal - compute fresh predictions
            specialist = primitive_set.get_specialist(tree)
            return specialist.predict(X)
        
        else:
            raise ValueError(f"Unknown tree node type: {type(tree)}")
    
    # ======================================================================
    # Copy
    # ======================================================================
    
    def copy(self) -> "EnsembleIndividual":
        """
        Create a deep copy of this ensemble.
        
        Returns
        -------
        EnsembleIndividual
            A new EnsembleIndividual with copied tree and metrics
        """
        tree_copy = self._deep_copy_tree(self.collection)
        
        new_ind = EnsembleIndividual(
            collection=tree_copy,
            primitive_set=self.primitive_set,
            depth=self.depth,
            nodes_count=self.nodes_count,
            total_nodes=self.total_nodes,
        )
        
        # Copy fitness metrics if they exist
        if self.fitness is not None:
            new_ind.fitness = self.fitness
        if self.test_fitness is not None:
            new_ind.test_fitness = self.test_fitness
        if self.errors_case is not None:
            new_ind.errors_case = self.errors_case.copy()
        
        # Copy slice indices
        new_ind._train_slice = self._train_slice
        new_ind._test_slice = self._test_slice
        
        return new_ind
    
    def _deep_copy_tree(self, tree: EnsembleTreeRepr) -> EnsembleTreeRepr:
        """Recursively deep copy the tree structure."""
        if isinstance(tree, tuple):
            condition = tree[0]
            if isinstance(condition, Condition):
                condition = condition.copy()
            return (
                condition,
                self._deep_copy_tree(tree[1]),
                self._deep_copy_tree(tree[2]),
            )
        else:
            return tree  # Strings are immutable
    
    # ======================================================================
    # String Representations
    # ======================================================================
    
    def __repr__(self) -> str:
        fitness_str = f", fitness={self.fitness:.6f}" if self.fitness is not None else ""
        return f"EnsembleIndividual(depth={self.depth}, nodes={self.total_nodes}{fitness_str})"
    
    def __str__(self) -> str:
        return self._tree_to_string(self.collection)
    
    def _tree_to_string(self, tree: EnsembleTreeRepr) -> str:
        """Convert tree to readable string format."""
        if isinstance(tree, tuple):
            condition = tree[0]
            cond_str = str(condition)
            true_str = self._tree_to_string(tree[1])
            false_str = self._tree_to_string(tree[2])
            return f"if({cond_str}, {true_str}, {false_str})"
        else:
            return str(tree)
    
    def get_tree_representation(self, indent: str = "") -> str:
        """Get a formatted string representation of the tree."""
        return self._format_tree(self.collection, indent)
    
    def _format_tree(self, tree: EnsembleTreeRepr, indent: str) -> str:
        """Format tree with indentation."""
        if isinstance(tree, tuple):
            condition = tree[0]
            result = f"{indent}if (\n"
            
            if isinstance(condition, Condition):
                result += condition.get_tree_representation(indent + "  ")
            else:
                result += f"{indent}  {condition}\n"
            
            result += f"{indent}) > 0 then\n"
            result += self._format_tree(tree[1], indent + "  ")
            result += f"{indent}else\n"
            result += self._format_tree(tree[2], indent + "  ")
            result += f"{indent}endif\n"
            return result
        else:
            return f"{indent}{tree}\n"
    
    def __copy__(self) -> "EnsembleIndividual":
        return self.copy()
    
    # ======================================================================
    # Tree Navigation Utilities
    # ======================================================================
    
    def get_specialist_indices(
        self,
        tree: Optional[EnsembleTreeRepr] = None,
        path: Optional[list] = None,
        depth: int = 1,
    ) -> list[tuple[list[int], int]]:
        """Get paths to all specialist terminals in the tree."""
        if tree is None:
            tree = self.collection
        if path is None:
            path = []
        
        indices = []
        if isinstance(tree, tuple):
            indices.extend(self.get_specialist_indices(tree[1], path + [1], depth + 1))
            indices.extend(self.get_specialist_indices(tree[2], path + [2], depth + 1))
        elif isinstance(tree, str):
            indices.append((path, depth))
        
        return indices
    
    def get_condition_indices(
        self,
        tree: Optional[EnsembleTreeRepr] = None,
        path: Optional[list] = None,
    ) -> list[list[int]]:
        """Get paths to all condition nodes in the tree."""
        if tree is None:
            tree = self.collection
        if path is None:
            path = []
        
        indices = []
        if isinstance(tree, tuple):
            indices.append(path + [0])
            indices.extend(self.get_condition_indices(tree[1], path + [1]))
            indices.extend(self.get_condition_indices(tree[2], path + [2]))
        
        return indices
    
    def get_subtree(self, path: list[int]) -> EnsembleTreeRepr:
        """Get the subtree at the given path."""
        element = self.collection
        for idx in path:
            element = element[idx]
        return element
    
    def replace_subtree(
        self,
        path: list[int],
        new_subtree: EnsembleTreeRepr,
    ) -> "EnsembleIndividual":
        """Create a new ensemble with a subtree replaced."""
        new_tree = self._replace_at_path(self.collection, path, new_subtree)
        return EnsembleIndividual(
            collection=new_tree,
            primitive_set=self.primitive_set,
        )
    
    def _replace_at_path(
        self,
        tree: EnsembleTreeRepr,
        path: list[int],
        new_subtree: EnsembleTreeRepr,
    ) -> EnsembleTreeRepr:
        """Replace subtree at path."""
        if not path:
            return new_subtree
        
        if isinstance(tree, tuple):
            index = path[0]
            tree_list = list(tree)
            tree_list[index] = self._replace_at_path(tree[index], path[1:], new_subtree)
            return tuple(tree_list)
        else:
            raise ValueError("Path leads into a terminal; cannot replace further.")
