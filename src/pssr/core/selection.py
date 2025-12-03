import random
import time
from typing import Callable, List, Tuple

import numpy as np

from pssr.core.misc import sample_without_replacement
from pssr.core.representations.individual import Individual
from pssr.core.representations.population import Population


# ---------------------------------- vectorized methods ---------------------------------- #
def tournament_min_vec(pool_size: int) -> Callable:
    def ts(pop, rng: np.random.Generator) -> List:
        n = len(pop.population) 
        # with replacement because pool_size is small compared to n
        choices = rng.choice(n, size=(n, pool_size), replace=True)
        
        fitness = pop.train_fitness
        pool_fitness = fitness[choices] 

        best_pos = np.argmin(pool_fitness, axis=1)
        
        winner_idx = choices[np.arange(n), best_pos]
        
        return [pop.population[i] for i in winner_idx]
    
    return ts


def eplex_min_vec(down_sampling: float = 1.0) -> Callable:
    def els(pop: Population, rng: np.random.Generator) -> List[Individual]:
        if pop.errors_case is None:
            raise ValueError("Errors case not calculated. Call calculate_errors_case() first.")
        errors = pop.errors_case
        N, M = errors.shape
        
        # number of selections is the same as individuals 
        S = N 
        
        med = np.median(errors, axis=0)
        mad = np.median(np.abs(errors - med), axis=0)
        
        n_cases = max(1, int(M * down_sampling))
        
        # case_order = rng.choice(M, size=(S, n_cases), replace=True)
        case_order = sample_without_replacement(rng, S, M, n_cases)  # (S, n_cases)

        # alive[s, i] = whether individual i is still in the pool of selection s
        alive = np.ones((S, N), dtype=bool)
        
        for t in range(n_cases): 
            c_t = case_order[:, t]
            E_t = errors[:, c_t].T
                        
            # threshold computation
            E_masked    = np.where(alive, E_t, np.inf)
            best_err    = E_masked.min(axis=1)
            eps_t       = mad[c_t]
            thr         = best_err + eps_t
            
            # update alive matrix
            new_alive   = E_t <= thr[:, None]
            alive       = alive & new_alive
        
        # select winners
        row_sums = alive.sum(axis=1, keepdims=True)
        empty    = (row_sums == 0).squeeze(1)
        if empty.any():
            alive[empty] = True
            row_sums = alive.sum(axis=1, keepdims=True)
            
        # convert alive mask into prob dist 
        weights = alive.astype(float) / row_sums
        
        cdf = np.cumsum(weights, axis=1)
        u   = rng.random(size=S)[:, None]
        winner_idx = (cdf >= u).argmax(axis=1)
        
        # Map to individuals
        winners = [pop.population[i] for i in winner_idx]
        return winners
    return els


def dalex_min_vec(down_sampling: float = 1.0, 
                  particularity_pressure: float = 20) -> Callable:
    def ds(pop: Population, rng: np.random.Generator) -> List[Individual]:
        # standardize the errors by case
        if pop.errors_case is None:
            raise ValueError("Errors case not calculated. Call calculate_errors_case() first.")
        pop.standardize_errors(std_errs=True)
        if pop.errors_case is None:
            raise ValueError("Errors case not standardized. Call standardize_errors() first.")
        
        errors = pop.errors_case    # (N, M)
        N, M = errors.shape
        S = N  # one selection per individual

        if down_sampling != 1.0:
            raise NotImplementedError("Vectorized DALex currently only for down_sampling == 1.0")

        # I_[s, j] ~ N(0, particularity_pressure)
        I_ = rng.normal(loc=0.0,
                        scale=particularity_pressure,
                        size=(S, M))          # (S, M)

        # softmax over cases → weights[s, j]
        I_ -= I_.max(axis=1, keepdims=True)
        exp_I = np.exp(I_)
        weights = exp_I / exp_I.sum(axis=1, keepdims=True)  # (S, M)

        # F[s, i] = Σ_j errors[i, j] * weights[s, j]
        F = weights @ errors.T    # (S, N)

        best_idx = np.argmin(F, axis=1)  # (S,)
        winners = [pop.population[i] for i in best_idx]
        return winners

    return ds
            

# ---------------------------------- standard methods ---------------------------------- #
def tournament_selection_min(pool_size: int) -> Callable:
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop): 
        """
        Selects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts

def tournament_selection_max(pool_size) -> Callable:
    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts


def double_tournament_min(pool_size) -> Callable:
    """
    Returns a function that performs tournament selection to select an individual with the lowest error and 
    size from a population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Double tournament selection: minimizes fitness (RMSE) first, then size.
        """
        pool = random.choices(pop.population, k=pool_size)

        # Sort by (fitness, total_nodes)
        # That means: primary objective is fitness, secondary is size
        best = min(pool, key=lambda x: (x.fitness, x.total_nodes))

        return best
    
    return ts 


def tournament_selection_min_size(pool_size, pressure_size=1e-4) -> Callable:
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness and size from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.
    pressure_size : float, optional
        Pressure for size in rank selection. Defaults to 1e-4.
    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness and size from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the combined lowest fitness and size in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the lowest fitness and size from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the combined lowest fitness and size in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness + pressure_size * ind.size for ind in pool])]

    return ts


def lexicase_selection_min(down_sampling=1.0) -> Callable:
    """
    Returns a function that performs lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    down_sampling : float, optional
        Proportion of test cases to sample. Defaults to 1.0

    Returns
    -------
    Callable
        A function ('ls') that performs lexicase selection on a population.
        
        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.

    Notes
    -----
    The returned function performs lexicase selection by receiving a population and returning the individual with the
    lowest fitness in the pool.
    """
    def ls(population) -> Tuple:
        """
        Perform lexicase selection on a population of individuals.
        
        Parameters
        ----------
        population : list of Individual
            The population from which to select parents.

        Returns
        -------
        Tuple [Individual, int]
            The selected parent individual and the number of cases used.
        """
        
        errors = population.errors_case
        num_cases = errors.shape[1]  
                
        pool = population.population.copy()
        n_cases = int(num_cases * down_sampling)  
        case_order = random.sample(range(num_cases), n_cases)
                        
        for i in range(n_cases):
            case_errors = errors[:, case_order[i]]
            
            best_individuals = np.where(case_errors == np.min(case_errors))[0]
                                                          
            if len(best_individuals) == 1:
                return pool[best_individuals[0]], i+1
            
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]
                    
        return random.choice(pool), i+1

    return ls 


def manual_epsilon_lexicase_selection_min(down_sampling=1.0, epsilon=1e-6) -> Callable:
    """
    Returns a function that performs manual epsilon lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    down_sampling : float, optional
        Proportion of test cases to sample. Defaults to 0.5.
    epsilon : float, optional
        The epsilon threshold for lexicase selection. Defaults to 1e-6.

    Returns
    -------
    Callable
        A function ('mels') that elects the individual with the lowest fitness in the pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.

    Notes
    -----
    The returned function performs lexicase selection by receiving a population and returning the individual with the
    lowest fitness in the pool.
    """
    def mels(pop) -> tuple:
        """
        Perform manual epsilon lexicase selection on a population of individuals.

        Parameters
        ----------
        pop : list of Individual
            The population from which to select parents.
        epsilon : float, optional
            The epsilon threshold for lexicase selection. Defaults to 1e-6.

        Returns
        -------
        Tuple [Individual, int]
            The selected parent individual and the number of cases used.
        """
        errors = pop.errors_case
        
        pool = pop.population.copy()
        num_cases = errors.shape[1]
        n_cases = int(num_cases * down_sampling) 
        case_order = random.sample(range(errors.shape[1]), n_cases) 

        for i in range(n_cases):
            case_idx = case_order[i] 
            case_errors = errors[:, case_idx]  # Get errors for this test case

            best_individuals = np.where(case_errors <= np.min(case_errors) + epsilon)[0]

            if len(best_individuals) == 1:
                return pool[best_individuals[0]], i+1
            
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]

        return random.choice(pool), i+1
    
    return mels


def epsilon_lexicase_selection_min(down_sampling=1.0) -> Callable:
    """
    Returns a function that performs epsilon lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    down_sampling : float, optional
        Proportion of test cases to sample. Defaults to 0.5.

    Returns
    -------
    Callable
        A function ('els') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.

    Notes
    -----
    The returned function performs lexicase selection by receiving a population and returning the individual with the
    lowest fitness in the pool.
    Epsilon is calculated with the median absolute deviation, as described in this paper: http://arxiv.org/abs/1905.13266
    The semi-dynamic version is implemented, which helps save computational power: http://arxiv.org/abs/1709.05394
    """

    def els(pop) -> Tuple:
        """
        Perform epsilon lexicase selection on a population of individuals.

        Parameters
        ----------
        pop : list of Individual
            The population from which to select parents.
        epsilon : float, optional
            The epsilon threshold for lexicase selection. Defaults to 1e-6.

        Returns
        -------
        Tuple [Individual, int]
            The selected parent individual and the number of cases used.
        """
        
        pop.calculate_mad() 
        errors = pop.errors_case
        
        num_cases = errors[0].shape[0]

        pool = pop.population.copy()
        # case_order = random.sample(range(num_cases), n_cases)  # ADDED

        n_cases = int(num_cases * down_sampling)
        for i in range(n_cases):
            # case_idx = case_order[i] 
            case_idx = random.choice(range(num_cases))
            case_errors = errors[:, case_idx] 

            # median_case = np.median(case_errors)
            # epsilon = np.median(np.abs(case_errors - median_case))  # Compute MAD for this case
            epsilon = pop.mad[case_idx]  # Get the MAD for this case

            # Get the best error on this test case across all individuals in the pool
            best_individuals = np.where(case_errors <= np.min(case_errors) + epsilon)[0]

            if len(best_individuals) == 1:
                return pool[best_individuals[0]], i+1
            
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]

        return random.choice(pool), i+1

    return els


def rank_based(mode='min', pool_size=2) -> Callable:
    """
    Returns a tournament function that performs rank-based selection to select an 
    individual with the lowest fitness and size from a population.

    Parameters
    ----------
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.
    pool_size : int, optional
        Number of individuals participating in the tournament. Defaults to 2.

    Returns
    -------
    Callable
        A function ('rs') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the combined lowest fitness and size in the pool.

    Notes
    -----
    The returned function performs rank-based selection by receiving a population and returning the individual with the
    lowest fitness 
    """

    if mode == 'max': 
        raise ValueError("Rank-based selection is only available for minimization problems.")

    def double_tournament(pop):
        """
        Perform rank-based selection on a population of individuals.

        Parameters
        ----------
        pop : list of Individual
            The population from which to select parents.

        Returns
        -------
        Individual
            The selected parent individual.
        """
        population, combined_ranks = pop.population, pop.combined_ranks
                
        selected_indices = np.random.choice(len(population), pool_size, replace=False)
        
        best_index = min(selected_indices, key=lambda idx: combined_ranks[idx])
        
        return population[best_index]
    
    return double_tournament


def dalex_min(down_sampling : float = 1.0, 
              particularity_pressure: float = 20) -> Callable:
    """
    Returns a function that performs DALex (Diversely Aggregated Lexicase Selection)
    to select an individual based on a weighted aggregation of test-case errors.

    Parameters
    ----------
    down_sampling : float, optional
        Proportion of test cases to sample. Defaults to 0.5.
    particularity_pressure : float, optional
        Standard deviation for the normal distribution used to sample importance scores.
        Higher values cause a more extreme weighting (more lexicase-like). Defaults to 20.
        
    Returns
    -------
    Callable
        A function that takes a population object and returns a tuple (selected individual, n_cases used).
    """

    def ds(pop):
        errors = pop.errors_case 
        num_total_cases = errors.shape[1]

        if down_sampling == 1:
            n_cases = num_total_cases
            subset_errors = errors

        else: 
            n_cases = int(num_total_cases * down_sampling)
            case_order = random.sample(range(num_total_cases), n_cases)
            subset_errors = errors[:, case_order]
            
        I_ = np.random.normal(0, particularity_pressure, size=n_cases)
        exp_I = np.exp(I_ - np.max(I_))
        weights = exp_I / np.sum(exp_I)
        F = np.dot(subset_errors, weights)

        best_index = np.argmin(F)
        return pop.population[best_index]

    return ds


def dalex_size_min(
                down_sampling=0.5, 
                particularity_pressure=20,
                tournament_size=2,
                p_best=0
                ) -> Callable:
    """
    Returns a function that performs DALex (Diversely Aggregated Lexicase Selection)
    to select an individual based on a weighted aggregation of test-case errors and then on a size tournament.

    Parameters
    ----------
    mode : str, optional
        'min' for minimization problems, 'max' for maximization problems. Defaults to 'min'.
    down_sampling : float, optional
        Proportion of test cases to sample. Defaults to 0.5.
    particularity_pressure : float, optional
        Standard deviation for the normal distribution used to sample importance scores.
        Higher values cause a more extreme weighting (more lexicase-like). Defaults to 20.
    tournament_size : int, optional
        Number of individuals participating in the size tournament. Defaults to 2.
    p_best : float, optional
            Probability of selecting the individual with the best fitness in the tournament. Defaults to 0.5.
            If p set to 0, then it is the dalex_selection_size vanilla version.

    Returns
    -------
    Callable
        A function that takes a population object and returns a tuple (selected individual, n_cases used).
    """

    def ds(pop):
        # Get the error matrix (assumed shape: (n_individuals, n_total_cases))
        errors = pop.errors_case.copy()
        num_total_cases = errors.shape[1]

        if down_sampling == 1:
            n_cases = num_total_cases
            subset_errors = errors

        else: 
            n_cases = int(num_total_cases * down_sampling)
            case_order = random.sample(range(num_total_cases), n_cases)
            subset_errors = errors[:, case_order]
        
        # Sample importance scores from N(0, particularity_pressure)
        I = np.random.normal(0, particularity_pressure, size=n_cases)
        exp_I = np.exp(I - np.max(I))
        weights = exp_I / np.sum(exp_I)
        F = np.dot(subset_errors, weights)

        if mode == 'min':
            if random.random() < p_best:
                best_index = np.argmin(F)
            else:
                sorted = np.argsort(F)
                best_index = sorted[:tournament_size]
                best_index = min(best_index, key=lambda idx: pop.population[idx].total_nodes)
        elif mode == 'max':
            if random.random() < p_best:
                best_index = np.argmax(F)
            else:
                sorted = np.argsort(F)[::-1]  
                best_index = sorted[:tournament_size]
                best_index = max(best_index, key=lambda idx: pop.population[idx].total_nodes)
        else:
            raise ValueError("Invalid mode. Use 'min' or 'max'.")
        
        # Return the selected individual and the number of cases used (n_cases)
        return pop.population[best_index]

    return ds


# --------------------------------- SPEED ENHANCER --------------------------------- #
class CaseSampler:
    def __init__(self, shape, cases):
        self.cases = cases
        self.indices = np.random.choice(shape, cases, replace=True)
        self.cursor = 0

    def sample(self, n_cases: int) -> np.ndarray:
        result = self.indices[self.cursor:self.cursor + n_cases]
        self.cursor = (self.cursor + n_cases) % self.cases
        return result
    
class PressureSampler:
    def __init__(self, mu: float, sigma: float, sample_size: int):
        self.pressures = np.random.normal(mu, sigma, sample_size)
        self.cursor = 0 

    def sample(self) -> int:
        result = self.pressures[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.pressures)
        return result


def dalex_fast_min(shape, 
                    n_cases=20,
                    **kwards):
    """
    Returns a function that performs a fast approxiamtion of DALex (Diversely Aggregated Lexicase Selection)
    to select an individual based on a weighted aggregation of test-case errors..

    Parameters
    ----------
    particularity_pressure : float, optional
        Standard deviation for the normal distribution used to sample importance scores.
        Higher values cause a more extreme weighting (more lexicase-like). Defaults to 20.

    Returns
    -------
    Callable
        A function that takes a population object and returns a tuple (selected individual, n_cases used).
    """
    sampler = CaseSampler(shape , 1_000_000)
    
    def ds(pop):
        errors = pop.errors_case
        idx = sampler.sample(n_cases)
        score = errors[:, idx].sum(axis=1)
        best_indices = np.argsort(score)[:2]
        best_index = random.choice(best_indices)
        return pop.population[best_index]
    return ds

def dalex_fast_min_rand(shape, 
                        particularity_pressure=10, 
                        tournament_size=2,
                        **kwargs):
    """
    Returns a fast, sparse approximation of DALEX selection using a pre-sampled
    list of 'particularity pressures' and a counter to avoid sampling every call.

    Parameters
    ----------
    particularity_pressure : float, optional
        Standard deviation for the normal distribution used to sample importance scores.
        Higher values cause a more extreme weighting (more lexicase-like). Defaults to 20.

    Returns
    -------
    ds : callable
        Function that takes a population object `pop` and returns
        (selected_individual, n_cases_used).
    """
    mu, sigma = get_musigma_from_cache(particularity_pressure)
    pressures = PressureSampler(mu, sigma, 10_000)
    sampler = CaseSampler(shape, 100_000)

    def ds(pop):
        errors = pop.errors_case
        n_cases = errors.shape[1]
        pp = pressures.sample()
        n = int(round(pp))
        n = max(1, min(n, n_cases))
        idx = sampler.sample(n)                
        score = errors[:, idx].sum(axis=1)
        best_indices = np.argsort(score)[:tournament_size]
        best_index = random.choice(best_indices)
        # best_index = np.argmin(score)
        return pop.population[best_index]
    return ds


def dalex_fast_min_size(particularity_pressure=20,
                                tournament_size=2,
                                p_best=1,
                                **kwards):
    """
    Returns a function that performs DALex (Diversely Aggregated Lexicase Selection)
    to select an individual based on a weighted aggregation of test-case errors and then on a size tournament.

    Parameters
    ----------
    particularity_pressure : float, optional
        Standard deviation for the normal distribution used to sample importance scores.
        Higher values cause a more extreme weighting (more lexicase-like). Defaults to 20.
    tournament_size : int, optional
        Number of individuals participating in the size tournament. Defaults to 2.
    p_best : float, optional
            Probability of selecting the individual with the best fitness in the tournament. Defaults to 0.5.
            If p set to 0, then it is the dalex_selection_size vanilla version.

    Returns
    -------
    Callable
        A function that takes a population object and returns a tuple (selected individual, n_cases used).
    """
  
    def ds(pop):
        errors = pop.errors_case 
        num_total_cases = errors.shape[1]
        idx = random.sample(range(num_total_cases), particularity_pressure)
        score = np.sum(errors[:,idx], axis=1)

        if random.random() < p_best:
            best_index = np.argmin(score)
        else: 
            sorted = np.argsort(score)
            best_index = sorted[:tournament_size]
            best_index = min(best_index, key=lambda idx: pop.population[idx].total_nodes)
        
        return pop.population[best_index]

    return ds


# ------------------------------- FETCHER -------------------------------- #
def fetch_selector(selector: str, 
                   problem_type: str = "min",
                   pool_size: int = 2,
                   down_sampling: float = 1.0,
                   particularity_pressure: float = 20,
                   **kwargs
                   ) -> Callable:
    
        if problem_type == "min":
            if selector == "tournament":
                return tournament_min_vec(pool_size)
            elif selector == "eplex":
                return eplex_min_vec(down_sampling=down_sampling)
            elif selector == "dalex":
                return dalex_min_vec(particularity_pressure=particularity_pressure)
            else: 
                raise ValueError(f"Unknown selector: {selector}")
        
        elif problem_type == "max":
            raise NotImplementedError("Only minimization selectors are implemented.")
        
        else: 
            raise ValueError(f"Unknown problem type: {problem_type}")




# ------------------------------- SPEED TESTING CODE ------------------------------- #
def benchmark_tournament(pop_size: int = 100,
                         pool_size: int = 2,
                         n_runs: int = 2_000) -> None:
    """Benchmark scalar vs vectorized tournament selection."""

    class Individual:
        def __init__(self, fitness, total_nodes):
            self.fitness = fitness
            self.total_nodes = total_nodes

    class Population:
        def __init__(self, fitness: np.ndarray):
            self.population = [
                Individual(fit, random.randint(1, 100)) for fit in fitness
            ]
            self.fitness = fitness

    # synthetic fitness values
    fitness = np.random.random(pop_size)
    pop = Population(fitness)

    ts_scalar = tournament_selection_min(pool_size)
    ts_vec = tournament_min_vec(pool_size)

    # scalar: pop_size selections per run
    t0 = time.perf_counter()
    for _ in range(n_runs):
        winners_scalar: List[Individual] = []
        for _ in range(pop_size):
            win_scalar = ts_scalar(pop)
            winners_scalar.append(win_scalar)
    t_scalar = time.perf_counter() - t0

    # vectorized: pop_size selections per run in one call
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        winners_vec = ts_vec(pop, rng)
    t_vec = time.perf_counter() - t0

    print("\n=== Tournament selection ===")
    print(f"Scalar   total: {t_scalar:.4f}s, per run: {t_scalar / n_runs:.6f}s")
    print(f"Vector   total: {t_vec:.4f}s, per run: {t_vec / n_runs:.6f}s")
    print(f"Scalar winners per run: {len(winners_scalar)}")
    print(f"Vector winners per run: {len(winners_vec)}")
    print(f"Speedup: {t_scalar / t_vec:.2f}x")


def benchmark_elexicase(pop_size: int = 1_000,
                        n_cases: int = 50,
                        down_sampling: float = 1.0,
                        n_runs: int = 5) -> None:
    """Benchmark scalar vs vectorized epsilon-lexicase selection."""

    class Individual:
        def __init__(self, idx: int):
            self.idx = idx

    class Population:
        def __init__(self, errors_case: np.ndarray):
            """
            errors_case: shape (pop_size, n_cases)
            """
            self.errors_case = errors_case
            self.population = [Individual(i) for i in range(errors_case.shape[0])]
            self.mad = None

        def calculate_mad(self):
            errors = self.errors_case
            med = np.median(errors, axis=0)
            self.mad = np.median(np.abs(errors - med), axis=0)

    errors_case = np.random.rand(pop_size, n_cases)
    pop = Population(errors_case)

    els_scalar = epsilon_lexicase_selection_min(down_sampling=down_sampling)
    els_vec = elexicase_min_vec(down_sampling=down_sampling)

    # scalar: pop_size selections per run
    t0 = time.perf_counter()
    for _ in range(n_runs):
        parents_scalar: List[Individual] = []
        for _ in range(pop_size):
            parent, _ = els_scalar(pop)
            parents_scalar.append(parent)
    t_scalar = time.perf_counter() - t0

    # vectorized: pop_size selections per run in one call
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        parents_vec = els_vec(pop, rng)
    t_vec = time.perf_counter() - t0

    print("\n=== Epsilon-lexicase selection ===")
    print(f"Scalar   total: {t_scalar:.4f}s, per run: {t_scalar / n_runs:.6f}s")
    print(f"Vector   total: {t_vec:.4f}s, per run: {t_vec / n_runs:.6f}s")
    print(f"Parents per run - scalar: {len(parents_scalar)}, vector: {len(parents_vec)}")
    print(f"Speedup: {t_scalar / t_vec:.2f}x")


def benchmark_dalex(pop_size: int = 1_000,
                    n_cases: int = 500,
                    down_sampling: float = 1.0,
                    particularity_pressure: float = 20.0,
                    n_runs: int = 50) -> None:
    """Benchmark scalar vs vectorized DALex selection."""

    class Individual:
        def __init__(self, idx: int):
            self.idx = idx

    class Population:
        def __init__(self, errors_case: np.ndarray):
            """
            errors_case: shape (pop_size, n_cases)
            """
            self.errors_case = errors_case
            self.population = [Individual(i) for i in range(errors_case.shape[0])]

    errors_case = np.random.rand(pop_size, n_cases)
    pop = Population(errors_case)

    ds_scalar = dalex_min(down_sampling=down_sampling,
                          particularity_pressure=particularity_pressure)
    ds_vec = dalex_min_vec(down_sampling=down_sampling,
                           particularity_pressure=particularity_pressure)

    # scalar: pop_size selections per run
    t0 = time.perf_counter()
    for _ in range(n_runs):
        parents_scalar: List[Individual] = []
        for _ in range(pop_size):
            parent = ds_scalar(pop)
            parents_scalar.append(parent)
    t_scalar = time.perf_counter() - t0

    # vectorized: pop_size selections per run in one call
    rng = np.random.default_rng(123)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        parents_vec = ds_vec(pop, rng)
    t_vec = time.perf_counter() - t0

    print("\n=== DALex selection ===")
    print(f"Scalar   total: {t_scalar:.4f}s, per run: {t_scalar / n_runs:.6f}s")
    print(f"Vector   total: {t_vec:.4f}s, per run: {t_vec / n_runs:.6f}s")
    print(f"Parents per run - scalar: {len(parents_scalar)}, vector: {len(parents_vec)}")
    print(f"Speedup: {t_scalar / t_vec:.2f}x")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # global reproducibility
    random.seed(42)
    np.random.seed(42)

    # benchmark_tournament()
    # benchmark_elexicase()
    benchmark_dalex()
