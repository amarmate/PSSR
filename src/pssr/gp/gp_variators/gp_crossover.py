from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from pssr.core.primitives import PrimitiveSet
from pssr.core.representations.individual import Individual
from pssr.gp.gp_initialization import PrimitiveCache
from pssr.gp.gp_variators.utils import (
    get_subtree,
    random_index_at_level,
    replace_subtree,
    sample_path,
    tree_depth,
    tree_depth_and_nodes,
)

CrossoverResult = Tuple[Individual, Individual]


def subtree_crossover(
    parent_a: Individual,
    parent_b: Individual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int,
    cache: PrimitiveCache | None = None,
    max_attempts: int = 10,
) -> CrossoverResult:
    """
    Perform subtree crossover between two parents using fast level-based sampling.
    
    Uses optimized approach from slim_gsgp_lib_np:
    - Selects level first, then finds valid crossover points
    - Uses fast iterative depth calculation
    - Avoids multiple attempts when possible
    
    Returns two offspring. If depth constraints cannot be satisfied, parents are copied.
    
    Parameters
    ----------
    parent_a : Individual
        First parent individual
    parent_b : Individual
        Second parent individual
    primitive_set : PrimitiveSet
        Primitive set for tree generation
    rng : np.random.Generator
        Random number generator
    max_depth : int
        Maximum depth constraint for offspring
    cache : PrimitiveCache, optional
        Cache for primitives (speeds up generation)
    max_attempts : int
        Maximum number of attempts to find valid crossover points
    Returns
    -------
    CrossoverResult
        Tuple of (offspring_a, offspring_b)
    """
    tree_a = parent_a.tree
    tree_b = parent_b.tree
    depth_a = parent_a.depth
    depth_b = parent_b.depth

    # Fast path: if both trees are terminals, no crossover possible
    if not isinstance(tree_a, tuple) or not isinstance(tree_b, tuple):
        return parent_a.copy(), parent_b.copy()

    for _ in range(max_attempts):
        # Fast level-based sampling: select level first, then node at that level
        level_a = int(rng.integers(0, depth_a))
        path_a = random_index_at_level(tree_a, level_a, rng)
        
        # Get subtree from parent A and calculate its depth
        subtree_a = get_subtree(tree_a, path_a)
        depth_subtree_a, _ = tree_depth_and_nodes(subtree_a)
        
        # Calculate maximum allowed level in parent B based on depth constraint
        max_level_b = min(max_depth - depth_subtree_a, depth_b - 1)
        if max_level_b < 0:
            continue  # Try again with different point in parent A
            
        # Select level in parent B that will satisfy depth constraint
        level_b = int(rng.integers(0, max_level_b + 1))
        path_b = random_index_at_level(tree_b, level_b, rng)
        
        # Get subtree from parent B and check depth constraint
        subtree_b = get_subtree(tree_b, path_b)
        depth_subtree_b, _ = tree_depth_and_nodes(subtree_b)
        
        # Check if crossover will satisfy depth constraints
        # depth_subtree_b + level_a <= max_depth ensures child_a is valid
        # depth_subtree_a + level_b <= max_depth ensures child_b is valid
        # Also ensure we're not swapping root with root (which would be no-op)
        if (depth_subtree_b + level_a <= max_depth and 
            depth_subtree_a + level_b <= max_depth and
            (level_b > 0 or depth_subtree_a > 1) and 
            (level_a > 0 or depth_subtree_b > 1)):
            
            child_a_tree = replace_subtree(tree_a, path_a, subtree_b)
            child_b_tree = replace_subtree(tree_b, path_b, subtree_a)
            
            return (
                Individual(child_a_tree, primitive_set=primitive_set),
                Individual(child_b_tree, primitive_set=primitive_set),
            )

    return parent_a.copy(), parent_b.copy()


def single_point_crossover(
    parent_a: Individual,
    parent_b: Individual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int,
    cache: PrimitiveCache | None = None,
) -> CrossoverResult:
    """
    Alias for subtree crossover.

    Traditional single-point crossover in tree GP is equivalent to swapping subtrees.
    
    Parameters
    ----------
    """
    return subtree_crossover(
        parent_a,
        parent_b,
        primitive_set,
        rng,
        max_depth,
        cache,
    )


CROSSOVER_REGISTRY: dict[str, Callable[..., CrossoverResult]] = {
    "subtree": subtree_crossover,
    "single_point": single_point_crossover,
}
