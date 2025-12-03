from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from pssr.core.primitives import GPFunction, PrimitiveSet
from pssr.gp.gp_initialization import PrimitiveCache

TreeRepr = tuple | str
TreePath = Tuple[int, ...]


def iter_paths(tree: TreeRepr, prefix: TreePath = ()) -> Iterable[tuple[TreePath, TreeRepr]]:
    """Yield all (path, subtree) pairs in depth-first order."""
    yield prefix, tree
    if isinstance(tree, tuple):
        for idx, child in enumerate(tree[1:]):
            yield from iter_paths(child, prefix + (idx,))


def iter_paths_with_level(
    tree: TreeRepr, prefix: TreePath = (), level: int = 0
) -> Iterable[tuple[TreePath, TreeRepr, int]]:
    """Yield all (path, subtree, level) tuples in depth-first order."""
    yield prefix, tree, level
    if isinstance(tree, tuple):
        for idx, child in enumerate(tree[1:]):
            yield from iter_paths_with_level(child, prefix + (idx,), level + 1)


def sample_path(
    tree: TreeRepr,
    rng: np.random.Generator,
    exclude_root: bool = False,
) -> TreePath:
    """Sample a random path (node) from the tree."""
    paths = [path for path, _ in iter_paths(tree)]
    if exclude_root and len(paths) > 1:
        paths = [path for path in paths if path]
    if not paths:
        return ()
    idx = rng.integers(0, len(paths))
    return paths[idx]


def sample_path_by_level(
    tree: TreeRepr,
    rng: np.random.Generator,
    exclude_root: bool = False,
) -> TreePath:
    """
    Sample a random path by first selecting a level uniformly, then a node at that level.
    
    This prevents bias toward leaves (which are more numerous) and gives balanced
    mutation across different tree depths, similar to slim_gsgp_lib_np.
    
    Optimized to calculate levels during iteration, avoiding extra passes.
    
    Parameters
    ----------
    tree : TreeRepr
        Tree to sample from
    rng : np.random.Generator
        Random number generator
    exclude_root : bool
        Whether to exclude the root node from sampling
        
    Returns
    -------
    TreePath
        Randomly selected path
    """
    # Build paths_by_level dict in a single pass
    paths_by_level: dict[int, list[TreePath]] = {}
    for path, _, level in iter_paths_with_level(tree):
        if exclude_root and level == 0:
            continue
        if level not in paths_by_level:
            paths_by_level[level] = []
        paths_by_level[level].append(path)
    
    if not paths_by_level:
        return ()
    
    # Sample a level uniformly (not weighted by number of nodes at that level)
    levels = list(paths_by_level.keys())
    selected_level = levels[rng.integers(0, len(levels))]
    
    # Sample uniformly from nodes at that level
    paths_at_level = paths_by_level[selected_level]
    idx = rng.integers(0, len(paths_at_level))
    return paths_at_level[idx]


def get_subtree(tree: TreeRepr, path: TreePath) -> TreeRepr:
    """Return the subtree located at ``path``."""
    node = tree
    for idx in path:
        if not isinstance(node, tuple):
            raise ValueError("Attempted to descend into terminal node")
        node = node[idx + 1]
    return node


def replace_subtree(tree: TreeRepr, path: TreePath, new_subtree: TreeRepr) -> TreeRepr:
    """Return a copy of ``tree`` where the subtree at ``path`` is replaced."""
    if not path:
        return new_subtree
    if not isinstance(tree, tuple):
        raise ValueError("Cannot replace subtree inside terminal")
    func_name = tree[0]
    children: list[TreeRepr] = list(tree[1:])
    child_idx = path[0]
    children[child_idx] = replace_subtree(children[child_idx], path[1:], new_subtree)
    return (func_name, *children)


def tree_depth(tree: TreeRepr) -> int:
    """Compute the depth of a tree."""
    if not isinstance(tree, tuple) or len(tree) == 0:
        return 1
    if len(tree) == 1:
        return 1
    return 1 + max(tree_depth(child) for child in tree[1:])


def tree_node_count(tree: TreeRepr) -> int:
    """Compute the total number of nodes in a tree."""
    if not isinstance(tree, tuple) or len(tree) == 0:
        return 1
    return 1 + sum(tree_node_count(child) for child in tree[1:])


def _sample_terminal_name(
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


def _sample_function(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    cache: PrimitiveCache | None,
    arity: int | None = None,
) -> GPFunction:
    if cache is not None and arity is None:
        return cache.get_function()
    functions = primitive_set.function_set_.functions
    if arity is not None:
        functions = [func for func in functions if func.arity == arity]
        if not functions:
            raise ValueError(f"No functions with arity={arity} available for point mutation.")
    idx = rng.integers(0, len(functions))
    return functions[idx]


def generate_random_tree(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int,
    cache: PrimitiveCache | None = None,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
) -> TreeRepr:
    """Generate a random tree up to ``max_depth`` using a Grow-like method."""
    if cache is not None:
        cache.reset_index()

    def _grow(depth: int, first_call: bool = True) -> TreeRepr:
        if depth <= 1 or (rng.random() < p_terminal and not first_call):
            return _sample_terminal_name(primitive_set, rng, cache, p_constant)
        gp_func = _sample_function(primitive_set, rng, cache)
        children = [_grow(depth - 1, first_call=False) for _ in range(gp_func.arity)]
        return (gp_func.name, *children)

    return _grow(max_depth)


def available_subtree_depth(max_depth: int, path: Sequence[int]) -> int:
    """
    Maximum depth allowed for a subtree located at ``path`` given ``max_depth``.

    Depth is measured in nodes, so replacing a node at depth ``len(path)+1`` leaves
    ``max_depth - len(path)`` levels available for the new subtree.
    """
    remaining = max_depth - len(path)
    return max(1, remaining)


def tree_depth_and_nodes(tree: TreeRepr) -> tuple[int, int]:
    """
    Fast iterative calculation of tree depth and node count using a stack.
    
    This is faster than recursive approaches, especially for deep trees.
    Adapted from slim_gsgp_lib_np for performance.
    
    Parameters
    ----------
    tree : TreeRepr
        Tree representation
        
    Returns
    -------
    tuple[int, int]
        (max_depth, total_nodes)
    """
    max_depth = 0
    total_nodes = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        total_nodes += 1
        if depth > max_depth:
            max_depth = depth
        if isinstance(node, tuple):
            for child in node[1:]:
                stack.append((child, depth + 1))
    return max_depth, total_nodes


def random_index_at_level(
    tree: TreeRepr,
    target_level: int,
    rng: np.random.Generator,
) -> TreePath:
    """
    Fast function to get a random index (path) at a specific level in the tree.
    
    This is faster than building full index dictionaries. It uses iterative
    traversal to find all nodes at the target level, then randomly selects one.
    Adapted from slim_gsgp_lib_np for performance, but uses 0-based indexing
    to match PSSR's convention.
    
    Parameters
    ----------
    tree : TreeRepr
        Tree to sample from
    target_level : int
        Target depth level (0 = root)
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    TreePath
        Random path to a node at the target level (0-based indices)
    """
    if target_level == 0:
        return ()
    
    # Use list instead of deque for type compatibility
    stack: list[tuple[TreeRepr, TreePath, int]] = [(tree, (), 0)]
    candidates: list[TreePath] = []
    
    while stack:
        node, path, level = stack.pop()
        if level == target_level:
            candidates.append(path)
        elif isinstance(node, tuple):
            # Use 0-based indexing to match PSSR's convention
            # enumerate(tree[1:]) gives indices 0, 1, 2, ... which map to tree[1], tree[2], tree[3], ...
            for i, child in enumerate(node[1:]):
                stack.append((child, path + (i,), level + 1))
    
    if not candidates:
        return ()
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


# -------------------------------- TESTING -------------------------------- #
if __name__ == "__main__":
    import time
    from collections import Counter

    import numpy as np

    from pssr.core.primitives import PrimitiveSet
    from pssr.gp.gp_initialization import grow_initializer

    print("=" * 60)
    print("Performance and Distribution Comparison")
    print("=" * 60)

    # Setup
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pset = PrimitiveSet(X, functions=["add", "mul", "sub", "div"], constant_range=1.0)
    rng = np.random.default_rng(0)

    # Generate test trees of varying sizes
    print("\nGenerating test trees...")
    trees = []
    for max_depth in [5, 7, 10]:
        individuals = grow_initializer(pset, rng, population_size=100, max_depth=max_depth)
        trees.extend([ind.tree for ind in individuals])
    
    print(f"Total test trees: {len(trees)}")
    print(f"Tree depths: {[tree_depth(t) for t in trees[:5]]}...\n")

    # Benchmark: Performance comparison
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    n_samples = 10000
    
    # Benchmark sample_path
    print(f"\nTesting sample_path ({n_samples} samples)...")
    start = time.perf_counter()
    paths_uniform = []
    for _ in range(n_samples):
        tree_idx = rng.integers(0, len(trees))
        tree = trees[tree_idx]
        path = sample_path(tree, rng, exclude_root=False)
        paths_uniform.append(len(path))  # Store level
    time_uniform = time.perf_counter() - start
    print(f"Time: {time_uniform*1000:.3f} ms")
    print(f"Average per sample: {time_uniform*1000/n_samples:.6f} ms")

    # Benchmark sample_path_by_level
    print(f"\nTesting sample_path_by_level ({n_samples} samples)...")
    start = time.perf_counter()
    paths_by_level = []
    for _ in range(n_samples):
        tree_idx = rng.integers(0, len(trees))
        tree = trees[tree_idx]
        path = sample_path_by_level(tree, rng, exclude_root=False)
        paths_by_level.append(len(path))  # Store level
    time_by_level = time.perf_counter() - start
    print(f"Time: {time_by_level*1000:.3f} ms")
    print(f"Average per sample: {time_by_level*1000/n_samples:.6f} ms")

    # Calculate overhead
    overhead = time_by_level / time_uniform if time_uniform > 0 else float('inf')
    overhead_ms = (time_by_level - time_uniform) * 1000
    overhead_pct = (overhead - 1) * 100 if overhead != float('inf') else 0

    print(f"\nPerformance Results:")
    print(f"  Overhead: {overhead:.2f}x ({overhead_pct:+.1f}%)")
    print(f"  Extra time per sample: {overhead_ms/n_samples:.6f} ms")
    print(f"  Total overhead: {overhead_ms:.3f} ms for {n_samples} samples")

    # Distribution analysis
    print("\n" + "=" * 60)
    print("Distribution Analysis")
    print("=" * 60)

    # Count level distributions
    uniform_dist = Counter(paths_uniform)
    by_level_dist = Counter(paths_by_level)

    # Get all levels present
    all_levels = sorted(set(list(uniform_dist.keys()) + list(by_level_dist.keys())))
    
    print(f"\nLevel distribution (out of {n_samples} samples):")
    print(f"{'Level':<8} {'Uniform':<12} {'By Level':<12} {'Difference':<12}")
    print("-" * 50)
    
    uniform_total = sum(uniform_dist.values())
    by_level_total = sum(by_level_dist.values())
    
    for level in all_levels:
        uniform_count = uniform_dist.get(level, 0)
        by_level_count = by_level_dist.get(level, 0)
        uniform_pct = uniform_count / uniform_total * 100 if uniform_total > 0 else 0
        by_level_pct = by_level_count / by_level_total * 100 if by_level_total > 0 else 0
        diff = by_level_pct - uniform_pct
        
        print(f"{level:<8} {uniform_pct:>6.2f}%     {by_level_pct:>6.2f}%     {diff:>+6.2f}%")

    # Statistical analysis
    print("\n" + "=" * 60)
    print("Statistical Summary")
    print("=" * 60)
    
    uniform_mean = np.mean(paths_uniform)
    by_level_mean = np.mean(paths_by_level)
    uniform_std = np.std(paths_uniform)
    by_level_std = np.std(paths_by_level)
    
    print(f"\nMean level selected:")
    print(f"  Uniform sampling: {uniform_mean:.3f} ± {uniform_std:.3f}")
    print(f"  By level sampling: {by_level_mean:.3f} ± {by_level_std:.3f}")
    print(f"  Difference: {by_level_mean - uniform_mean:+.3f}")

    print(f"\nStandard deviation:")
    print(f"  Uniform sampling: {uniform_std:.3f}")
    print(f"  By level sampling: {by_level_std:.3f}")
    print(f"  Difference: {by_level_std - uniform_std:+.3f}")

    # Impact on tree growth simulation
    print("\n" + "=" * 60)
    print("Impact on Tree Growth (Simulation)")
    print("=" * 60)
    
    # Simulate mutations: if we mutate at deeper levels, trees grow more
    # Count how often we select deep levels (>= 3)
    uniform_deep = sum(1 for level in paths_uniform if level >= 3)
    by_level_deep = sum(1 for level in paths_by_level if level >= 3)
    
    uniform_deep_pct = uniform_deep / len(paths_uniform) * 100
    by_level_deep_pct = by_level_deep / len(paths_by_level) * 100
    
    print(f"\nDeep level selection (level >= 3):")
    print(f"  Uniform sampling: {uniform_deep_pct:.2f}% ({uniform_deep}/{len(paths_uniform)})")
    print(f"  By level sampling: {by_level_deep_pct:.2f}% ({by_level_deep}/{len(paths_by_level)})")
    print(f"  Difference: {by_level_deep_pct - uniform_deep_pct:+.2f}%")
    
    # Count shallow levels (level <= 1)
    uniform_shallow = sum(1 for level in paths_uniform if level <= 1)
    by_level_shallow = sum(1 for level in paths_by_level if level <= 1)
    
    uniform_shallow_pct = uniform_shallow / len(paths_uniform) * 100
    by_level_shallow_pct = by_level_shallow / len(paths_by_level) * 100
    
    print(f"\nShallow level selection (level <= 1):")
    print(f"  Uniform sampling: {uniform_shallow_pct:.2f}% ({uniform_shallow}/{len(paths_uniform)})")
    print(f"  By level sampling: {by_level_shallow_pct:.2f}% ({by_level_shallow}/{len(paths_by_level)})")
    print(f"  Difference: {by_level_shallow_pct - uniform_shallow_pct:+.2f}%")

    print("\n" + "=" * 60)
    print("Conclusion:")
    print("=" * 60)
    print(f"1. Performance overhead: {overhead:.2f}x ({overhead_pct:+.1f}%)")
    print(f"2. By-level sampling reduces bias toward deep levels by {uniform_deep_pct - by_level_deep_pct:.1f}%")
    print(f"3. By-level sampling increases shallow mutations by {by_level_shallow_pct - uniform_shallow_pct:.1f}%")
    print("4. This helps control tree growth and provides more balanced exploration")
    print("=" * 60)

