import numpy as np
import pytest

from pssr.core.selection import (
    tournament_min_vec,
    eplex_min_vec,
    dalex_min_vec,
    epsilon_lexicase_selection_min,
    manual_epsilon_lexicase_selection_min,
    rank_based,
    fetch_selector,
)


class FakeIndividual:
    def __init__(self, fitness: float, total_nodes: int = 1):
        self.fitness = fitness
        self.total_nodes = total_nodes


class FakePopulation:
    def __init__(self, fitness: np.ndarray, errors_case: np.ndarray | None = None, sizes: np.ndarray | None = None):
        self.population = [FakeIndividual(fit, int(size) if sizes is not None else 1)
                           for fit, size in zip(fitness, sizes if sizes is not None else np.ones_like(fitness))]
        self.fitness = np.asarray(fitness)
        self.errors_case = errors_case if errors_case is not None else self._default_errors(len(fitness), 10)
        self.mad = None
        self.combined_ranks = np.argsort(np.argsort(self.fitness)).astype(float)

    @staticmethod
    def _default_errors(n_individuals: int, n_cases: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((n_individuals, n_cases))

    def calculate_mad(self):
        errors = self.errors_case
        med = np.median(errors, axis=0)
        self.mad = np.median(np.abs(errors - med), axis=0)


def test_tournament_min_vec_returns_one_per_individual():
    rng = np.random.default_rng(42)
    fitness = np.linspace(0.0, 1.0, 50)
    pop = FakePopulation(fitness)

    ts = tournament_min_vec(pool_size=3)
    winners = ts(pop, rng)

    assert isinstance(winners, list)
    assert len(winners) == len(pop.population)
    assert all(isinstance(w, FakeIndividual) for w in winners)


def test_eplex_min_vec_respects_shapes_and_returns_from_population():
    rng = np.random.default_rng(123)
    n, m = 40, 25
    fitness = np.linspace(0.0, 1.0, n)
    # errors_case shape (N, M)
    errors = np.abs(np.subtract.outer(fitness, np.linspace(0.0, 1.0, m)))
    pop = FakePopulation(fitness, errors)

    els = eplex_min_vec(down_sampling=1.0)
    winners = els(pop, rng)

    assert len(winners) == n
    assert set(map(id, winners)).issubset(set(map(id, pop.population)))


def test_dalex_min_vec_downsampling_not_implemented():
    rng = np.random.default_rng(9)
    fitness = np.linspace(0.0, 1.0, 10)
    pop = FakePopulation(fitness)

    ds = dalex_min_vec(down_sampling=0.5, particularity_pressure=5)
    with pytest.raises(NotImplementedError):
        _ = ds(pop, rng)


def test_fetch_selector_known_and_unknown():
    # Known selector returns a callable (vectorized version)
    sel = fetch_selector("dalex", problem_type="min", particularity_pressure=10)
    assert callable(sel)

    # Unknown selector raises informative error
    with pytest.raises(ValueError):
        _ = fetch_selector("unknown_strategy")

    # Unknown problem type raises
    with pytest.raises(ValueError):
        _ = fetch_selector("tournament", problem_type="weird")


def test_epsilon_and_manual_epsilon_lexicase_selection_min_api_compatibility():
    # Create population with errors favoring index 0 slightly
    n, m = 20, 15
    rng = np.random.default_rng(7)
    base_errors = rng.random((n, m))
    base_errors[0] = base_errors[0] * 0.5  # make individual 0 generally better

    fitness = rng.random(n)
    pop = FakePopulation(fitness, errors_case=base_errors)

    # Manual epsilon lexicase
    manual = manual_epsilon_lexicase_selection_min(down_sampling=1.0, epsilon=1e-6)
    parent_manual, used_cases_manual = manual(pop)
    assert isinstance(parent_manual, FakeIndividual)
    assert isinstance(used_cases_manual, int)

    # Epsilon lexicase uses MAD precompute
    eps = epsilon_lexicase_selection_min(down_sampling=1.0)
    parent_eps, used_cases_eps = eps(pop)
    assert isinstance(parent_eps, FakeIndividual)
    assert isinstance(used_cases_eps, int)


def test_rank_based_min_raises_on_max_and_selects_valid_individual():
    with pytest.raises(ValueError):
        _ = rank_based(mode='max')

    fitness = np.array([0.9, 0.1, 0.5, 0.3])
    pop = FakePopulation(fitness)

    selector = rank_based(mode='min', pool_size=2)
    selected = selector(pop)
    assert isinstance(selected, FakeIndividual)
    assert selected in pop.population
