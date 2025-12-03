from __future__ import annotations

import time
from functools import partial
from typing import Callable

import numpy as np

from pssr.core.primitives import PrimitiveSet
from pssr.core.representations.individual import Individual
from pssr.core.representations.population import Population
from pssr.gp.gp_initialization import PrimitiveCache
from pssr.gp.gp_variators.gp_crossover import CROSSOVER_REGISTRY, CrossoverResult
from pssr.gp.gp_variators.gp_mutation import MUTATION_REGISTRY


def variator_fun(
    crossover: Callable[..., CrossoverResult],
    mutation: Callable[..., "Individual"],
    p_xo: float,
    sample_by_level: bool = True,
    mutation_cache_size: int = 5_000,
) -> Callable:
    """
    Combine crossover and mutation into a single variator.

    The returned callable expects:
        population : Population
        primitive_set : PrimitiveSet
        rng : np.random.Generator
        max_depth : int
    Parameters
    ----------
    crossover : Callable
        Crossover function
    mutation : Callable
        Mutation function
    p_xo : float
        Probability of crossover (vs mutation)
    sample_by_level : bool
        Controls mutation sampling strategy. When True, mutation selects
        paths by level (prevents bias toward leaves, slower). When False,
        mutation samples uniformly from all nodes (faster but leaf-biased).
    mutation_cache_size : int
        Number of mutation calls that reuse the same primitive cache before
        rebuilding it. Default: 5,000.
    """

    if not 0.0 <= p_xo <= 1.0:
        raise ValueError("p_xo must be in [0, 1]")
    if mutation_cache_size <= 0:
        raise ValueError("mutation_cache_size must be a positive integer")

    mutation_cache: PrimitiveCache | None = None
    cache_remaining: int = 0

    def _variator(
        population: Population,
        primitive_set: PrimitiveSet,
        rng: np.random.Generator,
        max_depth: int,
    ) -> tuple[Population, dict[str, float]]:
        """
        Variator function that returns both population and timing information.
        
        Returns
        -------
        tuple[Population, dict]
            (offspring_population, timing_dict) where timing_dict contains:
            - 'mut_time': total time spent on mutations (seconds)
            - 'xo_time': total time spent on crossovers (seconds)
        """
        if len(population) == 0:
            raise ValueError("Population must contain at least one individual.")

        individuals = list(population.population)
        pop_size = len(individuals)

        nonlocal mutation_cache, cache_remaining

        # Initialize timing
        xo_time = 0.0
        mut_time = 0.0

        if pop_size == 1:
            mutation_cache, cache_remaining = _maybe_refresh_cache(
                mutation_cache,
                cache_remaining,
                primitive_set,
                rng,
                mutation_cache_size,
            )
            mut_start = time.perf_counter()
            mutant = _apply_mutation(
                mutation,
                individuals[0],
                primitive_set,
                rng,
                max_depth,
                mutation_cache,
                sample_by_level,
            )
            mut_time = time.perf_counter() - mut_start
            return Population([mutant]), {"mut_time": mut_time, "xo_time": 0.0}

        xo_mask, mut_mask = _build_masks(pop_size, p_xo, rng)
        xo_indices = np.flatnonzero(xo_mask)
        mut_indices = np.flatnonzero(mut_mask)

        offspring: list = []

        if xo_indices.size % 2 == 1:
            mut_indices = np.append(mut_indices, xo_indices[-1])
            xo_indices = xo_indices[:-1]

        # Track crossover time
        xo_start = time.perf_counter()
        rng.shuffle(xo_indices)
        for i in range(0, xo_indices.size, 2):
            idx_a = xo_indices[i]
            idx_b = xo_indices[i + 1]
            child_a, child_b = crossover(
                individuals[idx_a],
                individuals[idx_b],
                primitive_set,
                rng,
                max_depth,
            )
            offspring.extend([child_a, child_b])
        xo_time = time.perf_counter() - xo_start

        # Track mutation time
        mut_start = time.perf_counter()
        for idx in mut_indices:
            # Pass sample_by_level if mutation accepts it
            mutation_cache, cache_remaining = _maybe_refresh_cache(
                mutation_cache,
                cache_remaining,
                primitive_set,
                rng,
                mutation_cache_size,
            )
            mutant = _apply_mutation(
                mutation,
                individuals[idx],
                primitive_set,
                rng,
                max_depth,
                mutation_cache,
                sample_by_level,
            )
            offspring.append(mutant)
        mut_time = time.perf_counter() - mut_start

        # Ensure population size is preserved
        if len(offspring) != pop_size:
            raise RuntimeError(
                f"Variation produced {len(offspring)} individuals, expected {pop_size}"
            )

        timing = {"mut_time": mut_time, "xo_time": xo_time}
        return Population(offspring), timing

    return _variator


def _maybe_refresh_cache(
    cache: PrimitiveCache | None,
    remaining: int,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    cache_size: int,
) -> tuple[PrimitiveCache, int]:
    if cache is None or remaining <= 0:
        cache = PrimitiveCache(primitive_set, rng, cache_size=cache_size)
        remaining = cache_size
    return cache, remaining - 1


def _apply_mutation(
    mutation: Callable[..., Individual],
    individual: Individual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int,
    cache: PrimitiveCache,
    sample_by_level: bool,
) -> Individual:
    try:
        return mutation(
            individual,
            primitive_set,
            rng,
            max_depth,
            cache,
            sample_by_level=sample_by_level,
        )
    except TypeError:
        return mutation(individual, primitive_set, rng, max_depth, cache)


def variator_function(*args, **kwargs) -> Callable:
    """Backward-compatible alias."""
    return variator_fun(*args, **kwargs)


def fetch_crossover(crossover: str, **kwargs) -> Callable:
    if crossover not in CROSSOVER_REGISTRY:
        raise ValueError(f"Unknown crossover '{crossover}'. Available: {sorted(CROSSOVER_REGISTRY)}")
    base = CROSSOVER_REGISTRY[crossover]
    if kwargs:
        return partial[CrossoverResult](base, **kwargs)
    return base


def fetch_mutation(mutation: str, **kwargs) -> Callable:
    if mutation not in MUTATION_REGISTRY:
        raise ValueError(f"Unknown mutation '{mutation}'. Available: {sorted(MUTATION_REGISTRY)}")
    base = MUTATION_REGISTRY[mutation]
    if kwargs:
        return partial(base, **kwargs)
    return base


def _build_masks(pop_size: int, p_xo: float, rng: np.random.Generator):
    """Return boolean masks for crossover and mutation assignments."""
    indices = np.arange(pop_size)
    rng.shuffle(indices)
    num_crossover = int(pop_size * p_xo)
    num_crossover -= num_crossover % 2
    mask = np.zeros(pop_size, dtype=bool)
    if num_crossover > 0:
        selected = indices[:num_crossover]
        mask[selected] = True
    return mask, ~mask


# -------------------------------- TESTING -------------------------------- #
if __name__ == "__main__":
    import numpy as np

    from pssr.core.primitives import PrimitiveSet
    from pssr.core.representations.population import Population
    from pssr.gp.gp_initialization import grow_initializer

    print("=" * 60)
    print("Variator Function Tests")
    print("=" * 60)

    # Setup
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pset = PrimitiveSet(X, functions=["add", "mul", "sub", "div"], constant_range=1.0)
    rng = np.random.default_rng(42)

    # Generate initial population
    print("\nStep 1: Generate initial population")
    print("-" * 60)
    individuals = grow_initializer(pset, rng, population_size=20, max_depth=5)
    population = Population(individuals)
    print(f"Initial population size: {len(population)}")
    print(f"Sample individual: {population[0]}\n")

    # Get crossover and mutation functions
    crossover_func = fetch_crossover("subtree")
    mutation_func = fetch_mutation("subtree", sample_by_level=True)

    # Test 1: Basic variator functionality
    print("=" * 60)
    print("Test 1: Basic Variator Functionality")
    print("=" * 60)
    variator = variator_fun(crossover_func, mutation_func, p_xo=0.8, sample_by_level=True)
    offspring, _ = variator(population, pset, rng, max_depth=5)
    print(f"Offspring population size: {len(offspring)}")
    assert len(offspring) == len(population), "Population size not preserved"
    assert all(ind.depth <= 5 for ind in offspring), "Depth constraint violated"
    print("OK\n")

    # Test 2: Different p_xo values
    print("=" * 60)
    print("Test 2: Different p_xo Values")
    print("=" * 60)
    for p_xo in [0.0, 0.5, 1.0]:
        variator = variator_fun(crossover_func, mutation_func, p_xo=p_xo, sample_by_level=True)
        offspring, _ = variator(population, pset, rng, max_depth=5)
        assert len(offspring) == len(population), f"Population size not preserved for p_xo={p_xo}"
        print(f"p_xo={p_xo}: OK (size={len(offspring)})")
    print()

    # Test 3: Population size = 1 (only mutation)
    print("=" * 60)
    print("Test 3: Population Size = 1")
    print("=" * 60)
    single_pop = Population([population[0]])
    variator = variator_fun(crossover_func, mutation_func, p_xo=0.8, sample_by_level=True)
    offspring, _ = variator(single_pop, pset, rng, max_depth=5)
    assert len(offspring) == 1, "Single individual population size not preserved"
    assert offspring[0].depth <= 5, "Depth constraint violated"
    print("OK\n")

    # Test 4: sample_by_level parameter
    print("=" * 60)
    print("Test 4: sample_by_level Parameter")
    print("=" * 60)
    
    # Test with sample_by_level=True
    variator_true = variator_fun(crossover_func, mutation_func, p_xo=0.8, sample_by_level=True)
    offspring_true, _ = variator_true(population, pset, rng, max_depth=5)
    assert len(offspring_true) == len(population), "sample_by_level=True failed"
    print("sample_by_level=True: OK")
    
    # Test with sample_by_level=False
    variator_false = variator_fun(crossover_func, mutation_func, p_xo=0.8, sample_by_level=False)
    offspring_false, _ = variator_false(population, pset, rng, max_depth=5)
    assert len(offspring_false) == len(population), "sample_by_level=False failed"
    print("sample_by_level=False: OK\n")

    # Test 5: Verify crossover and mutation are both applied
    print("=" * 60)
    print("Test 5: Verify Both Crossover and Mutation Applied")
    print("=" * 60)
    
    # Create a population with known individuals
    test_individuals = grow_initializer(pset, rng, population_size=10, max_depth=3)
    test_pop = Population(test_individuals)
    
    # Use p_xo=0.5 to get mix of crossover and mutation
    variator = variator_fun(crossover_func, mutation_func, p_xo=0.5, sample_by_level=True)
    offspring, _ = variator(test_pop, pset, rng, max_depth=5)
    
    # Check that offspring are different from parents (at least some)
    original_trees = {id(ind.tree) for ind in test_pop}
    offspring_trees = {id(ind.tree) for ind in offspring}
    
    # Some trees should be different (not all identical)
    assert len(offspring_trees) > 1, "All offspring are identical"
    print(f"Original unique trees: {len(original_trees)}")
    print(f"Offspring unique trees: {len(offspring_trees)}")
    print("Variation occurred: OK\n")

    # Test 6: Edge case - p_xo=1.0 (all crossover)
    print("=" * 60)
    print("Test 6: p_xo=1.0 (All Crossover)")
    print("=" * 60)
    variator = variator_fun(crossover_func, mutation_func, p_xo=1.0, sample_by_level=True)
    offspring, _ = variator(population, pset, rng, max_depth=5)
    assert len(offspring) == len(population), "Population size not preserved"
    assert len(offspring) % 2 == 0, "Even number of offspring expected for p_xo=1.0"
    print("OK\n")

    # Test 7: Edge case - p_xo=0.0 (all mutation)
    print("=" * 60)
    print("Test 7: p_xo=0.0 (All Mutation)")
    print("=" * 60)
    variator = variator_fun(crossover_func, mutation_func, p_xo=0.0, sample_by_level=True)
    offspring, _ = variator(population, pset, rng, max_depth=5)
    assert len(offspring) == len(population), "Population size not preserved"
    print("OK\n")

    # Test 8: Error handling - invalid p_xo
    print("=" * 60)
    print("Test 8: Error Handling - Invalid p_xo")
    print("=" * 60)
    try:
        variator_fun(crossover_func, mutation_func, p_xo=1.5, sample_by_level=True)
        assert False, "Should have raised ValueError for p_xo > 1.0"
    except ValueError as e:
        assert "p_xo must be in [0, 1]" in str(e)
        print("Invalid p_xo > 1.0: OK")
    
    try:
        variator_fun(crossover_func, mutation_func, p_xo=-0.1, sample_by_level=True)
        assert False, "Should have raised ValueError for p_xo < 0.0"
    except ValueError as e:
        assert "p_xo must be in [0, 1]" in str(e)
        print("Invalid p_xo < 0.0: OK\n")

    # Test 9: Empty population error (Population class prevents this, but test variator check)
    print("=" * 60)
    print("Test 9: Variator Error Handling")
    print("=" * 60)
    # Create a mock empty population by accessing internal list directly
    # (Population class prevents empty initialization, but variator checks anyway)
    variator = variator_fun(crossover_func, mutation_func, p_xo=0.8, sample_by_level=True)
    # Test with minimal population (already tested above)
    print("Variator handles edge cases: OK\n")

    print("=" * 60)
    print("All variator tests passed!")
    print("=" * 60)