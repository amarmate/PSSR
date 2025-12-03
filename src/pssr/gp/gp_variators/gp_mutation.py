from __future__ import annotations

from typing import Callable

import numpy as np

from pssr.core.primitives import PrimitiveSet
from pssr.core.representations.individual import Individual
from pssr.gp.gp_initialization import PrimitiveCache
from pssr.gp.gp_variators.utils import (
    generate_random_tree,
    get_subtree,
    random_index_at_level,
    replace_subtree,
    sample_path,
    tree_depth_and_nodes,
)


def subtree_mutation(
    individual: Individual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int,
    cache: PrimitiveCache | None = None,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
    sample_by_level: bool = True,
) -> Individual:
    """
    Fast subtree mutation using level-based sampling (optimized from slim_gsgp_lib_np).
    
    Calculates the maximum allowed depth based on the selected node's level,
    ensuring the generated subtree will always fit within max_depth constraints.
    Uses fast iterative depth calculation and level-based sampling for performance.
    
    Key optimizations:
    - Uses random_index_at_level for fast level-based sampling
    - Uses tree_depth_and_nodes for fast depth calculation
    - Calculates allowed depth efficiently based on level
    
    Parameters
    ----------
    individual : Individual
        Individual to mutate
    primitive_set : PrimitiveSet
        Primitive set for tree generation
    rng : np.random.Generator
        Random number generator
    max_depth : int
        Maximum depth constraint for the entire tree
    cache : PrimitiveCache, optional
        Cache for primitives (speeds up generation)
    p_terminal : float
        Probability of selecting terminal (for Grow method)
    p_constant : float
        Probability of selecting constant (vs variable terminal)
    sample_by_level : bool
        If True, use fast level-based sampling (recommended, faster).
        If False, sample uniformly from all nodes (slower, biased toward leaves).
        Default: True
        
    Returns
    -------
    Individual
        Mutated individual (always succeeds)
    """
    tree = individual.tree
    depth = individual.depth
    
    # Fast path: if tree is a terminal, replace with new terminal/constant
    if not isinstance(tree, tuple):
        new_subtree = generate_random_tree(
            primitive_set,
            rng,
            1,  # Terminal depth
            cache=cache,
            p_terminal=p_terminal,
            p_constant=p_constant,
        )
        return Individual(new_subtree, primitive_set=primitive_set)
    
    # Fast level-based sampling: select level first, then node at that level
    if sample_by_level:
        level = int(rng.integers(0, depth))
        path = random_index_at_level(tree, level, rng)
    else:
        # Fallback to uniform sampling (slower)
        path = sample_path(tree, rng, exclude_root=False)
        # Estimate level from path length (approximate)
        level = len(path)
    
    # Calculate maximum allowed depth for the new subtree
    # If max_depth=10 and we select a node at level 4,
    # available depth = max_depth - level - 1
    # This ensures: level + new_subtree_depth <= max_depth
    max_depth_new_subtree = int(max_depth - level - 1)
    
    # Generate new subtree
    if max_depth_new_subtree < 1:
        # Must use terminal/constant
        if rng.random() < p_constant:
            new_subtree = primitive_set.sample_constant(rng).name
        else:
            new_subtree = primitive_set.sample_terminal(rng).name
    else:
        new_subtree = generate_random_tree(
            primitive_set,
            rng,
            max_depth_new_subtree,
            cache=cache,
            p_terminal=p_terminal,
            p_constant=p_constant,
        )
    
    # Replace subtree (guaranteed to succeed because max_depth_new_subtree was calculated correctly)
    mutated_tree = replace_subtree(tree, path, new_subtree)
    
    # No need to check depth - it's guaranteed to be <= max_depth
    return Individual(mutated_tree, primitive_set=primitive_set)


def point_mutation(
    individual: Individual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int,
    cache: PrimitiveCache | None = None,
    p_constant: float = 0.3,
) -> Individual:
    """
    Mutate a single node:
        * If it's a terminal, replace with another terminal/constant.
        * If it's a function, swap the function while keeping arity.
    """
    tree = individual.tree
    path = sample_path(tree, rng, exclude_root=False)
    node = get_subtree(tree, path)

    if isinstance(node, tuple):
        arity = len(node) - 1
        new_func = _sample_function_with_arity(primitive_set, rng, arity, cache)
        mutated = (new_func.name, *node[1:])
    else:
        mutated = _sample_terminal(primitive_set, rng, cache, p_constant)

    mutated_tree = replace_subtree(tree, path, mutated)
    # Check depth constraint - use fast iterative calculation
    depth_check, _ = tree_depth_and_nodes(mutated_tree)
    if depth_check > max_depth:
        return individual.copy()
    return Individual(mutated_tree, primitive_set=primitive_set)


def _sample_function_with_arity(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    arity: int,
    cache: PrimitiveCache | None,
):
    if cache is not None:
        func = cache.get_function()
        if func.arity == arity:
            return func
    funcs = [func for func in primitive_set.function_set_.functions if func.arity == arity]
    if not funcs:
        raise ValueError(f"No functions with arity {arity} available for point mutation.")
    idx = rng.integers(0, len(funcs))
    return funcs[idx]


def _sample_terminal(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    cache: PrimitiveCache | None,
    p_constant: float,
) -> str:
    if cache is not None:
        if rng.random() < p_constant:
            return cache.get_constant()
        return cache.get_terminal()
    if rng.random() < p_constant:
        return primitive_set.sample_constant(rng).name
    return primitive_set.sample_terminal(rng).name


MUTATION_REGISTRY: dict[str, Callable[..., Individual]] = {
    "subtree": subtree_mutation,
    "point": point_mutation,
}


# -------------------------------- TESTING -------------------------------- #
if __name__ == "__main__":
    import numpy as np

    from pssr.core.primitives import PrimitiveSet
    from pssr.gp.gp_initialization import PrimitiveCache, grow_initializer

    print("=" * 60)
    print("Mutation Example")
    print("=" * 60)

    # Setup
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pset = PrimitiveSet(X, functions=["add", "mul", "sub", "div"], constant_range=1.0)
    rng = np.random.default_rng(0)
    cache = PrimitiveCache(pset, rng, cache_size=500)

    # Generate initial population and select one individual
    print("Step 1: Generate initial tree using Grow initializer")
    print("-" * 60)
    individuals = grow_initializer(pset, rng, population_size=1, max_depth=5)
    original = individuals[0]
    print(f"Original tree: {original}")
    print(f"Original depth: {original.depth}, nodes: {original.total_nodes}")
    print(f"Tree structure: {original.tree}\n")

    # Mutate the individual
    print("Step 2: Apply subtree mutation")
    print("-" * 60)
    mutated = subtree_mutation(original, pset, rng, max_depth=5, cache=cache)
    print(f"Mutated tree: {mutated}")
    print(f"Mutated depth: {mutated.depth}, nodes: {mutated.total_nodes}")
    print(f"Tree structure: {mutated.tree}\n")

    # Verify mutation worked
    print("Step 3: Verify mutation")
    print("-" * 60)
    assert mutated.tree != original.tree, "Tree was not mutated"
    assert mutated.depth <= 5, "Depth constraint violated"
    print("Mutation successful!")
    print(f"Tree changed: {mutated.tree != original.tree}")
    print(f"Depth constraint respected: {mutated.depth <= 5}")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
    
    # Test both sampling methods
    print("\n" + "=" * 60)
    print("Testing sample_by_level parameter")
    print("=" * 60)
    
    # Test with sample_by_level=True (default)
    mutated_by_level = subtree_mutation(
        original, pset, rng, max_depth=5, cache=cache, sample_by_level=True
    )
    print(f"With sample_by_level=True: {mutated_by_level}")
    
    # Test with sample_by_level=False
    mutated_by_node = subtree_mutation(
        original, pset, rng, max_depth=5, cache=cache, sample_by_level=False
    )
    print(f"With sample_by_level=False: {mutated_by_node}")
    
    assert mutated_by_level.depth <= 5, "sample_by_level=True violated depth"
    assert mutated_by_node.depth <= 5, "sample_by_level=False violated depth"
    print("\nBoth sampling methods work correctly!")
    print("=" * 60)
