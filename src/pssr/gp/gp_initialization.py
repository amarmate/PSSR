from typing import Callable

import numpy as np

from pssr.core.primitives import GPFunction, PrimitiveSet
from pssr.core.representations.individual import Individual

TreeRepr = tuple | str  # Tree representation: tuple for functions, str for terminals


class PrimitiveCache:
    """
    Cache for pre-sampled primitives to speed up tree generation and mutations.
    
    Pre-samples terminals, constants, and functions to avoid repeated method calls
    and list indexing operations during tree creation.
    
    Attributes
    ----------
    terminals : list[str]
        Cached terminal names
    constants : list[str]
        Cached constant names
    functions : list[GPFunction]
        Cached function objects
    cache_size : int
        Size of the cache
    _idx : int
        Current index in the cache (for cycling)
    """
    
    def __init__(
        self,
        primitive_set: PrimitiveSet,
        rng: np.random.Generator,
        cache_size: int = 5_000,
    ):
        """
        Initialize the cache by pre-sampling primitives.
        
        Parameters
        ----------
        primitive_set : PrimitiveSet
            Primitive set to sample from
        rng : np.random.Generator
            Random number generator
        cache_size : int
            Number of items to pre-sample for each primitive type
        """
        self.cache_size = cache_size
        self._idx = 0
        
        # Pre-sample terminals
        self.terminals = [
            primitive_set.sample_terminal(rng).name
            for _ in range(cache_size)
        ]
        
        # Pre-sample constants
        self.constants = [
            primitive_set.sample_constant(rng).name
            for _ in range(cache_size)
        ]
        
        # Pre-sample functions
        self.functions = [
            primitive_set.sample_function(rng)
            for _ in range(cache_size)
        ]
    
    def get_terminal(self) -> str:
        """Get next terminal from cache (cycles through cache)."""
        idx = self._idx % self.cache_size
        self._idx += 1
        return self.terminals[idx]
    
    def get_constant(self) -> str:
        """Get next constant from cache (cycles through cache)."""
        idx = self._idx % self.cache_size
        self._idx += 1
        return self.constants[idx]
    
    def get_function(self) -> GPFunction:
        """Get next function from cache (cycles through cache)."""
        idx = self._idx % self.cache_size
        self._idx += 1
        return self.functions[idx]
    
    def reset_index(self) -> None:
        """Reset the cache index (useful for starting a new tree)."""
        self._idx = 0


def _create_grow_tree(
    depth: int,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    cache: PrimitiveCache,
    p_t: float = 0.5,
    p_c: float = 0.3,
    first_call: bool = True,
) -> TreeRepr:
    """
    Create a random tree using the Grow method with cached primitives.
    
    In the Grow method, nodes are randomly selected from functions and terminals
    until the maximum depth is reached. At maximum depth, only terminals are selected.
    
    Parameters
    ----------
    depth : int
        Maximum depth of the tree
    primitive_set : PrimitiveSet
        Primitive set containing functions, terminals, and constants
    rng : np.random.Generator
        Random number generator
    cache : PrimitiveCache
        Cache of pre-sampled primitives for fast access
    p_t : float
        Probability of selecting a terminal (vs function) when depth > 1
    p_c : float
        Probability of selecting a constant (vs variable terminal) when selecting terminal
    first_call : bool
        Whether this is the root call (always creates a function at root)
        
    Returns
    -------
    TreeRepr
        Tree representation as tuple or string
    """
    # At maximum depth or randomly choose terminal (but not at root)
    if (depth <= 1 or rng.random() < p_t) and not first_call:
        if rng.random() > p_c:
            # Choose variable terminal from cache
            return cache.get_terminal()
        else:
            # Choose constant from cache
            return cache.get_constant()
    
    # Choose a function from cache
    gp_func = cache.get_function()
    arity = gp_func.arity
    
    # Recursively create children
    children = [
        _create_grow_tree(depth - 1, primitive_set, rng, cache, p_t, p_c, first_call=False)
        for _ in range(arity)
    ]
    
    return (gp_func.name, *children)


def _create_full_tree(
    depth: int,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    cache: PrimitiveCache,
    p_c: float = 0.3,
) -> TreeRepr:
    """
    Create a random tree using the Full method with cached primitives.
    
    In the Full method, all nodes at depth < max_depth are functions,
    and only nodes at max_depth are terminals. This creates complete trees.
    
    Parameters
    ----------
    depth : int
        Maximum depth of the tree
    primitive_set : PrimitiveSet
        Primitive set containing functions, terminals, and constants
    rng : np.random.Generator
        Random number generator
    cache : PrimitiveCache
        Cache of pre-sampled primitives for fast access
    p_c : float
        Probability of selecting a constant (vs variable terminal) at leaves
        
    Returns
    -------
    TreeRepr
        Tree representation as tuple or string
    """
    # At maximum depth, choose terminal from cache
    if depth <= 1:
        if rng.random() > p_c:
            return cache.get_terminal()
        else:
            return cache.get_constant()
    
    # Choose a function from cache (all internal nodes are functions in Full method)
    gp_func = cache.get_function()
    arity = gp_func.arity
    
    # Recursively create children
    children = [
        _create_full_tree(depth - 1, primitive_set, rng, cache, p_c)
        for _ in range(arity)
    ]
    
    return (gp_func.name, *children)


def grow_initializer(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    population_size: int,
    max_depth: int,
    init_depth: int | None = None,
    p_t: float = 0.5,
    p_c: float = 0.2,
    cache_size: int = 500,
) -> list[Individual]:
    """
    Initialize population using the Grow method with cached primitives.
    
    All trees are created using the Grow method with the specified max_depth.
    Uses a cache of pre-sampled primitives for faster tree generation.
    
    Parameters
    ----------
    primitive_set : PrimitiveSet
        Primitive set for tree generation
    rng : np.random.Generator
        Random number generator
    population_size : int
        Number of individuals to create
    max_depth : int
        Maximum depth of trees
    init_depth : int, optional
        Not used in grow method (kept for API compatibility)
    p_t : float
        Probability of selecting terminal vs function
    p_c : float
        Probability of selecting constant vs variable terminal
    cache_size : int
        Size of the primitive cache (default: 500)
        
    Returns
    -------
    list[Individual]
        List of Individual objects
    """
    cache = PrimitiveCache(primitive_set, rng, cache_size)
    individuals = []
    for _ in range(population_size):
        cache.reset_index()  # Reset index for each new tree
        tree = _create_grow_tree(max_depth, primitive_set, rng, cache, p_t, p_c)
        individual = Individual(tree=tree, primitive_set=primitive_set)
        individuals.append(individual)
    
    return individuals


def full_initializer(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    population_size: int,
    max_depth: int,
    init_depth: int | None = None,
    p_c: float = 0.3,
    cache_size: int = 500,
) -> list[Individual]:
    """
    Initialize population using the Full method with cached primitives.
    
    All trees are created using the Full method, creating complete trees
    of the specified max_depth. Uses a cache of pre-sampled primitives.
    
    Parameters
    ----------
    primitive_set : PrimitiveSet
        Primitive set for tree generation
    rng : np.random.Generator
        Random number generator
    population_size : int
        Number of individuals to create
    max_depth : int
        Maximum depth of trees
    init_depth : int, optional
        Not used in full method (kept for API compatibility)
    p_c : float
        Probability of selecting constant vs variable terminal at leaves
    cache_size : int
        Size of the primitive cache (default: 500)
        
    Returns
    -------
    list[Individual]
        List of Individual objects
    """
    cache = PrimitiveCache(primitive_set, rng, cache_size)
    individuals = []
    for _ in range(population_size):
        cache.reset_index()  # Reset index for each new tree
        tree = _create_full_tree(max_depth, primitive_set, rng, cache, p_c)
        individual = Individual(tree=tree, primitive_set=primitive_set)
        individuals.append(individual)
    
    return individuals


def rhh_initializer(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    population_size: int,
    max_depth: int,
    init_depth: int = 2,
    p_t: float = 0.5,
    p_c: float = 0.3,
    cache_size: int = 500,
) -> list[Individual]:
    """
    Initialize population using Ramped Half-and-Half (RHH) method with cached primitives.
    
    RHH creates trees with depths ranging from init_depth to max_depth.
    For each depth, half the trees use the Full method and half use the Grow method.
    This creates diversity in both tree depth and structure.
    Uses a cache of pre-sampled primitives for faster tree generation.
    
    Parameters
    ----------
    primitive_set : PrimitiveSet
        Primitive set for tree generation
    rng : np.random.Generator
        Random number generator
    population_size : int
        Number of individuals to create
    max_depth : int
        Maximum depth of trees
    init_depth : int
        Minimum depth of trees (default: 2)
    p_t : float
        Probability of selecting terminal vs function (for Grow trees)
    p_c : float
        Probability of selecting constant vs variable terminal
    cache_size : int
        Size of the primitive cache (default: 500)
        
    Returns
    -------
    list[Individual]
        List of Individual objects
    """
    if init_depth < 2:
        init_depth = 2
    
    if max_depth < init_depth:
        max_depth = init_depth
    
    cache = PrimitiveCache(primitive_set, rng, cache_size)
    individuals = []
    depth_range = max_depth - init_depth + 1
    inds_per_depth = population_size / depth_range
    
    # Create trees for each depth level
    for curr_depth in range(init_depth, max_depth + 1):
        n_full = int(inds_per_depth // 2)
        n_grow = int(inds_per_depth // 2)
        
        # Full method trees
        for _ in range(n_full):
            cache.reset_index()
            tree = _create_full_tree(curr_depth, primitive_set, rng, cache, p_c)
            individual = Individual(tree=tree, primitive_set=primitive_set)
            individuals.append(individual)
        
        # Grow method trees
        for _ in range(n_grow):
            cache.reset_index()
            tree = _create_grow_tree(curr_depth, primitive_set, rng, cache, p_t, p_c)
            individual = Individual(tree=tree, primitive_set=primitive_set)
            individuals.append(individual)
    
    # Fill remaining slots with grow trees at max_depth
    while len(individuals) < population_size:
        cache.reset_index()
        tree = _create_grow_tree(max_depth, primitive_set, rng, cache, p_t, p_c)
        individual = Individual(tree=tree, primitive_set=primitive_set)
        individuals.append(individual)
    
    return individuals


def fetch_initializer(
    initializer: str = "rhh",
    init_depth: int = 2,
    max_depth: int = 6,
    p_t: float = 0.5,
    p_c: float = 0.3,
    cache_size: int = 500,
    **kwargs,
) -> Callable:
    """
    Factory function to get an initializer function with caching support.
    
    Parameters
    ----------
    initializer : str
        Name of initializer method: "rhh", "grow", or "full"
    init_depth : int
        Minimum depth for RHH method
    max_depth : int
        Maximum depth for all methods
    p_t : float
        Terminal probability for Grow/RHH methods
    p_c : float
        Constant probability for all methods
    cache_size : int
        Size of the primitive cache (default: 500)
    **kwargs
        Additional arguments (passed to initializer)
        
    Returns
    -------
    Callable
        Initializer function with signature:
        (primitive_set, rng, population_size, max_depth, **kwargs) -> list[Individual]
    """
    if initializer == "rhh":
        def initializer_func(primitive_set, rng, population_size, max_depth, **kw):
            return rhh_initializer(
                primitive_set, rng, population_size, max_depth,
                init_depth=init_depth, p_t=p_t, p_c=p_c,
                cache_size=cache_size
            )
        return initializer_func
    
    elif initializer == "grow":
        def initializer_func(primitive_set, rng, population_size, max_depth, **kw):
            return grow_initializer(
                primitive_set, rng, population_size, max_depth,
                init_depth=init_depth, p_t=p_t, p_c=p_c,
                cache_size=cache_size
            )
        return initializer_func
    
    elif initializer == "full":
        def initializer_func(primitive_set, rng, population_size, max_depth, **kw):
            return full_initializer(
                primitive_set, rng, population_size, max_depth,
                init_depth=init_depth, p_c=p_c,
                cache_size=cache_size
            )
        return initializer_func
    
    else:
        raise ValueError(f"Unknown initializer: {initializer!r}. Choose from: 'rhh', 'grow', 'full'")


# -------------------------------- TESTING -------------------------------- #
if __name__ == "__main__":
    import numpy as np

    from pssr.core.primitives import PrimitiveSet
    
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pset = PrimitiveSet(X, functions=["add", "mul", "sub"], constant_range=1.0)
    rng = np.random.default_rng(42)
    
    print("=" * 60)
    print("Test 1: Grow Initializer")
    print("=" * 60)
    individuals = grow_initializer(pset, rng, population_size=10, max_depth=3)
    assert len(individuals) == 10
    assert all(isinstance(ind, Individual) for ind in individuals)
    assert all(ind.depth <= 3 for ind in individuals)
    print("OK\n")
    
    print("=" * 60)
    print("Test 2: Full Initializer")
    print("=" * 60)
    individuals = full_initializer(pset, rng, population_size=10, max_depth=3)
    assert len(individuals) == 10
    print("OK\n")
    
    print("=" * 60)
    print("Test 3: RHH Initializer")
    print("=" * 60)
    individuals = rhh_initializer(pset, rng, population_size=20, max_depth=4, init_depth=2)
    assert len(individuals) == 20
    assert all(2 <= ind.depth <= 4 for ind in individuals)
    print("OK\n")
    
    print("=" * 60)
    print("Test 4: Factory Function")
    print("=" * 60)
    init_func = fetch_initializer("rhh", init_depth=2, max_depth=3)
    individuals = init_func(pset, rng, population_size=10, max_depth=3)
    assert len(individuals) == 10
    
    # Test cache reuse
    individuals1 = init_func(pset, rng, population_size=5, max_depth=3)
    individuals2 = init_func(pset, rng, population_size=5, max_depth=3)
    assert len(individuals1) == len(individuals2) == 5
    
    # Test invalid initializer
    try:
        _ = fetch_initializer("invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("OK\n")
    
    print("=" * 60)
    print("Test 5: Cache Functionality")
    print("=" * 60)
    cache = PrimitiveCache(pset, rng, cache_size=100)
    assert len(cache.terminals) == 100
    assert len(cache.constants) == 100
    assert len(cache.functions) == 100
    
    initial_idx = cache._idx
    cache.get_terminal()
    assert cache._idx == initial_idx + 1
    cache.reset_index()
    assert cache._idx == 0
    print("OK\n")
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
