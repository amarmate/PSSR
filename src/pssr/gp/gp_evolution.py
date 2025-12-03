
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Callable

import numpy as np
import numpy.typing as npt

from pssr.core.primitives import PrimitiveSet
from pssr.core.representations.individual import Individual
from pssr.core.representations.population import Population
from pssr.gp.gp_utils import LogDict, init_log, print_verbose_report, update_log

Array = npt.NDArray[np.float64]


def GPevo(
    population: Population | Sequence[Individual],
    X: Array,
    y: Array,
    primitive_set: PrimitiveSet,
    selector: Callable,
    variator: Callable,
    rng: np.random.Generator,
    n_generations: int,
    max_depth: int,
    train_slice: slice,
    test_slice: slice | None = None,
    elitism: int = 1,
    verbose: int = 0,
) -> tuple[Population, Individual, LogDict]:
    """
    Main Genetic Programming evolutionary loop.

    Parameters
    ----------
    population : Population | Sequence[Individual]
        Initial population (Population object or list of individuals).
    X : Array
        Combined inputs (train + test), shape (n_total_samples, n_features).
    y : Array
        Combined targets (train + test), shape (n_total_samples,).
    primitive_set : PrimitiveSet
        Primitive set used for evaluation and new tree generation.
    selector : Callable
        Selection operator. Must accept (population, rng) or (population,)
        and return either a list of individuals or a Population.
    variator : Callable
        Variation operator returned by `variator_fun`.
    rng : np.random.Generator
        Random number generator for reproducibility.
    train_slice : slice
        Slice indices for training portion of X and y.
    test_slice : slice | None
        Slice indices for testing portion of X and y. If None, no testing data is used.
    n_generations : int
        Number of evolutionary generations to run.
    max_depth : int
        Maximum allowed depth for individuals (passed to variator).
    elitism : int
        Number of top individuals to copy directly to the next generation.
    verbose : int, default=0
        Verbosity level. If > 0, prints progress report after each generation.

    Returns
    -------
    tuple[Population, Individual, dict]
        Final population, best individual found, and log dictionary with
        generation statistics.
    """
    if n_generations < 0:
        raise ValueError(f"n_generations must be >= 0, got {n_generations}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    if elitism < 0:
        raise ValueError(f"elitism must be >= 0, got {elitism}")

    population_obj = Population.from_sequence(population)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}"
        )

    # Initialize log with test fitness support if test data provided
    has_test_data = test_slice is not None
    log = init_log(include_test=has_test_data)
    best_fitness = np.inf
    cumulative_time = 0.0
    
    # Precompute semantics for initial population (train + optional test)
    population_obj.set_slices(train_slice, test_slice)
    population_obj.calculate_semantics(X)
    population_obj.calculate_errors_case(y)

    # Evaluate on training set (semantics already computed)
    eval_start = time.perf_counter()
    metrics = population_obj.fit_and_assess(X, y)
    eval_time = time.perf_counter() - eval_start
    cumulative_time += eval_time
    
    update_log(log, generation=0, metrics=metrics, eval_time=eval_time)

    best_individual: Individual = metrics["best_individual"].copy()
    best_fitness = metrics["best_fitness"]
    elites = population_obj.extract_elites(elitism)

    # Verbose reporting for generation 0
    if verbose > 0:
        # Try to get selector name (handles closures from factory functions)
        selector_name = None
        if hasattr(selector, "__name__"):
            selector_name = selector.__name__
        elif hasattr(selector, "__qualname__"):
            # For closures, try to extract name from qualname
            qualname = selector.__qualname__
            if "." in qualname:
                selector_name = qualname.split(".")[-1]
        
        print_verbose_report(
            generation=0,
            metrics=metrics,
            eval_time=eval_time,
            population=population_obj,
            X=X[train_slice],
            gen_time=cumulative_time,
            selector_name=selector_name,
            first=True,
        )

    if n_generations == 0:
        return population_obj, best_individual, log

    for generation in range(1, n_generations + 1):
        gen_start = time.perf_counter()
        parents = _apply_selector(selector, population_obj, rng)
        offspring, timing_info = variator(parents, primitive_set, rng, max_depth)

        if elitism > 0 and elites:
            offspring.inject_elites(elites, rng)

        # Recompute semantics for offspring (train + optional test)
        offspring.set_slices(train_slice, test_slice)
        offspring.calculate_semantics(X)
        offspring.calculate_errors_case(y)

        eval_start = time.perf_counter()
        metrics = offspring.fit_and_assess(X, y)
        eval_time = time.perf_counter() - eval_start
        gen_time = time.perf_counter() - gen_start
        cumulative_time += gen_time

        population_obj = offspring
        update_log(log, generation=generation, metrics=metrics, eval_time=eval_time)

        if metrics["best_fitness"] < best_fitness:
            best_fitness = metrics["best_fitness"]
            best_individual = metrics["best_individual"].copy()

        elites = population_obj.extract_elites(elitism)

        # Verbose reporting for each generation
        if verbose > 0:
            # Try to get selector name (handles closures from factory functions)
            selector_name = None
            if hasattr(selector, "__name__"):
                selector_name = selector.__name__
            elif hasattr(selector, "__qualname__"):
                # For closures, try to extract name from qualname
                qualname = selector.__qualname__
                if "." in qualname:
                    selector_name = qualname.split(".")[-1]
            
            print_verbose_report(
                generation=generation,
                metrics=metrics,
                eval_time=eval_time,
                population=population_obj,
                X=X[train_slice],
                gen_time=cumulative_time,
                mut_time=timing_info.get("mut_time"),
                xo_time=timing_info.get("xo_time"),
                selector_name=selector_name,
                lex_rounds=None,
                first=False,
            )

    return population_obj, best_individual, log


def _apply_selector(
    selector: Callable,
    population: Population,
    rng: np.random.Generator,
) -> Population:
    try:
        selection = selector(population, rng)
    except TypeError:
        selection = selector(population)

    if isinstance(selection, tuple):
        selection = selection[0]

    if isinstance(selection, Population):
        selected_pop = selection
    elif isinstance(selection, list):
        selected_pop = Population(selection)
    else:
        raise TypeError(
            "Selector must return a Population or list of Individuals."
        )

    if len(selected_pop) != len(population):
        raise ValueError(
            "Selector must return the same number of individuals as the input population."
        )

    return selected_pop


if __name__ == "__main__":
    import numpy as np

    from pssr.core.primitives import PrimitiveSet
    from pssr.core.selection import fetch_selector
    from pssr.gp.gp_initialization import fetch_initializer
    from pssr.gp.gp_variation import fetch_crossover, fetch_mutation, variator_fun

    print("=" * 60)
    print("GPevo Test Suite")
    print("=" * 60)

    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = 2 * X.squeeze() + 0.1 * rng.normal(size=X.shape[0])

    pset = PrimitiveSet(X, functions=["add", "sub", "mul"], constant_range=1.0)

    # Test 1: Basic GPevo execution
    print("\nTest 1: Basic GPevo Execution")
    print("-" * 60)
    initializer = fetch_initializer("rhh", init_depth=2, max_depth=4, cache_size=500)
    individuals = initializer(pset, rng, population_size=20, max_depth=4)
    population = Population(individuals)

    selector = fetch_selector("tournament", pool_size=3)
    crossover = fetch_crossover("single_point")
    mutation = fetch_mutation("subtree")
    variator = variator_fun(crossover, mutation, p_xo=0.8)

    final_pop, best_ind, log = GPevo(
        population=population,
        X=X,
        y=y,
        primitive_set=pset,
        selector=selector,
        variator=variator,
        rng=rng,
        n_generations=3,
        max_depth=4,
        train_slice=slice(0, X.shape[0]),
    )

    assert len(final_pop) == len(population), "Population size changed"
    assert best_ind is not None, "Best individual is None"
    assert best_ind.fitness is not None, "Best individual has no fitness"
    assert best_ind.fitness >= 0, "Fitness should be non-negative"
    assert len(log["generation"]) == 4, "Log should have 4 entries (gen 0 + 3 generations)"
    assert log["best_fitness"][-1] == best_ind.fitness, "Log best fitness doesn't match"
    print("OK: Basic execution works\n")

    # Test 2: Population evaluation methods
    print("Test 2: Population Evaluation Methods")
    print("-" * 60)
    test_pop = Population(final_pop.population[:5])  # Use subset for speed
    
    # Test evaluate / fitness calculation
    test_pop.set_slices(slice(0, X.shape[0]))
    test_pop.calculate_semantics(X)
    test_pop.evaluate(X, y)
    assert test_pop.train_fitness is not None, "Fitness not calculated"
    assert test_pop.train_fitness.shape == (5,), "Fitness shape incorrect"
    assert all(f >= 0 for f in test_pop.train_fitness), "Fitness values should be non-negative"
    assert all(ind.fitness is not None for ind in test_pop), "Individual fitness not set"
    
    # Test calculate_errors_case
    test_pop.calculate_errors_case(y)
    assert test_pop.errors_case is not None, "errors_case not calculated"
    assert test_pop.errors_case.shape == (5, len(X)), "errors_case shape incorrect"
    assert all(ind.errors_case is not None for ind in test_pop), "Individual errors_case not set"
    
    # Test get_best_individual
    best = test_pop.get_best_individual()
    assert best.fitness == test_pop.train_fitness.min(), "Best individual fitness incorrect"
    print("OK: Population methods work correctly\n")

    # Test 3: n_generations=0 (no evolution)
    print("Test 3: n_generations=0 (No Evolution)")
    print("-" * 60)
    initial_pop = Population(individuals[:10])
    final_pop_zero, best_zero, log_zero = GPevo(
        population=initial_pop,
        X=X,
        y=y,
        primitive_set=pset,
        selector=selector,
        variator=variator,
        rng=rng,
        n_generations=0,
        max_depth=4,
        train_slice=slice(0, X.shape[0]),
    )
    assert len(final_pop_zero) == len(initial_pop), "Population size changed"
    assert len(log_zero["generation"]) == 1, "Log should have 1 entry (gen 0 only)"
    assert log_zero["generation"][0] == 0, "Log generation should start at 0"
    print("OK: n_generations=0 works\n")

    # Test 4: Elitism
    print("Test 4: Elitism")
    print("-" * 60)
    initial_pop_elite = Population(individuals[:15])
    final_pop_elite, best_elite, log_elite = GPevo(
        population=initial_pop_elite,
        X=X,
        y=y,
        primitive_set=pset,
        selector=selector,
        variator=variator,
        rng=rng,
        n_generations=2,
        max_depth=4,
        train_slice=slice(0, X.shape[0]),
        elitism=2,
    )
    assert len(final_pop_elite) == len(initial_pop_elite), "Population size changed"
    assert best_elite.fitness is not None, "Best individual has no fitness"
    print("OK: Elitism works\n")

    # Test 5: Log structure and content
    print("Test 5: Log Structure and Content")
    print("-" * 60)
    assert "generation" in log, "Log missing 'generation'"
    assert "best_fitness" in log, "Log missing 'best_fitness'"
    assert "mean_fitness" in log, "Log missing 'mean_fitness'"
    assert "worst_fitness" in log, "Log missing 'worst_fitness'"
    assert "std_fitness" in log, "Log missing 'std_fitness'"
    assert "best_depth" in log, "Log missing 'best_depth'"
    assert "best_size" in log, "Log missing 'best_size'"
    assert "eval_time" in log, "Log missing 'eval_time'"
    
    assert len(log["best_fitness"]) == len(log["generation"]), "Log lengths don't match"
    assert all(isinstance(t, float) for t in log["eval_time"]), "eval_time should be floats"
    assert all(t >= 0 for t in log["eval_time"]), "eval_time should be non-negative"
    assert all(f >= 0 for f in log["best_fitness"]), "best_fitness should be non-negative"
    print("OK: Log structure is correct\n")

    # Test 6: Fitness improvement (or at least non-degradation in early gens)
    print("Test 6: Fitness Tracking")
    print("-" * 60)
    # Best fitness should be tracked correctly (may not always improve, but should be tracked)
    best_fitnesses = log["best_fitness"]
    assert len(best_fitnesses) > 0, "No fitness values logged"
    assert all(isinstance(f, float) for f in best_fitnesses), "Fitness values should be floats"
    assert all(f >= 0 for f in best_fitnesses), "Fitness values should be non-negative"
    print(f"Initial best fitness: {best_fitnesses[0]:.4f}")
    print(f"Final best fitness: {best_fitnesses[-1]:.4f}")
    print("OK: Fitness tracking works\n")

    # Test 7: Different selectors
    print("Test 7: Different Selectors")
    print("-" * 60)
    for sel_name in ["tournament", "eplex", "dalex"]:
        try:
            test_sel = fetch_selector(sel_name, pool_size=2)
            test_pop_sel = Population(individuals[:10])
            _, _, _ = GPevo(
                population=test_pop_sel,
                X=X,
                y=y,
                primitive_set=pset,
                selector=test_sel,
                variator=variator,
                rng=rng,
                n_generations=1,
                max_depth=4,
                train_slice=slice(0, X.shape[0]),
            )
            print(f"  {sel_name}: OK")
        except Exception as e:
            print(f"  {sel_name}: FAILED - {e}")
            raise
    print()

    # Test 8: Verbose reporting
    print("=" * 60)
    print("Test 8: Verbose Reporting")
    print("=" * 60)
    test_pop_verbose = Population(individuals[:100])
    _, _, _ = GPevo(
        population=test_pop_verbose,
        X=X,
        y=y,
        primitive_set=pset,
        selector=selector,
        variator=variator,
        rng=rng,
        n_generations=100,
        max_depth=7,
        train_slice=slice(0, X.shape[0]),
        verbose=1,
    )
    print("OK: Verbose reporting works\n")

    print("=" * 60)
    print("All GPevo tests passed!")
    print("=" * 60)