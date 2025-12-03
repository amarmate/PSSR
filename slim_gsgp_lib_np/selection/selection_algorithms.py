# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Selection operator implementation.
"""

import random
import numpy as np
from slim_gsgp_lib_np.selection.utils import get_musigma_from_cache

def selector(problem='min', 
             type='tournament', 
             pool_size=2, 
             down_sampling=5,
             particularity_pressure=20,
             epsilon=1e-6, 
             dalex_size_prob=0.5,
             n_cases=2,
             n=1000,
             ):
    """
    Returns a selection function based on the specified problem and selection type.

    Parameters
    ----------
    problem : str, optional
        The type of problem to solve. Can be 'min' or 'max'. Defaults to 'min'.
    type : str, optional
        The type of selection to perform. Can be 'tournament', 'e_lexicase', 'lexicase', 'manual_e_lexicase', 
        'roulette', 'rank_based', 'tournament_size', or 'dalex'. Defaults to 'tournament'.
    pool_size : int, optional
        Number of individuals participating in the tournament. Defaults to 2.
    down_sampling : float, optional
        Down sampling rate for lexicase selection. Defaults to 5.
    particularity_pressure : float, optional
        Standard deviation used in DALex for sampling importance scores. Defaults to 20.
    epsilon : float, optional
        Epsilon value used in epsilon lexicase selection. Defaults to 1e-6.
    dalex_size_prob : float, optional
        Probability of selecting the individual with the best fitness in the tournament. Defaults to 0.5.
    n_cases : int, optional
        Number of cases to sample for DALex selection. Defaults to 2.

    Returns
    -------
    Callable
        A selection function that selects an individual from a population based on the specified problem and selection type.
    """

    if problem not in ('min', 'max'):
        raise ValueError(f"Invalid problem type: {problem}")
    mode = problem

    if type == 'double_tournament' and pool_size <= 2:
        print("Warning: pool_size must be >2 for double tournament. Setting to 3.")
        pool_size = 3

    pp = particularity_pressure
    ds = down_sampling

    MODED = {
        'tournament':        lambda: tournament_selection_min(pool_size) if mode=='min'
                                    else tournament_selection_max(pool_size),
        'double_tournament': lambda: double_tournament_min(pool_size) if mode=='min'
                                    else None, # No max version of double tournament
        'e_lexicase':        lambda: epsilon_lexicase_selection(mode=mode, down_sampling=ds),
        'manual_e_lexicase': lambda: manual_epsilon_lexicase_selection(mode=mode, down_sampling=ds, epsilon=epsilon),
        'lexicase':          lambda: lexicase_selection(mode=mode, down_sampling=ds),
        'dalex':             lambda: dalex_min(down_sampling=ds, particularity_pressure=pp),
        'rank_based':        lambda: rank_based(mode=mode, pool_size=pool_size),
        'dalex_size':        lambda: dalex_size(mode=mode, down_sampling=ds, particularity_pressure=pp, tournament_size=pool_size, p_best=dalex_size_prob),
        'dalex_fast':        lambda: dalex_fast_min(n, n_cases=n_cases, tournament_size=pool_size),
        'dalex_fast_rand':   lambda: dalex_fast_min_rand(n, particularity_pressure=pp, tournament_size=pool_size),
        'dalex_fast_size':   lambda: dalex_fast_min_size(particularity_pressure=pp, p_best=dalex_size_prob, tournament_size=pool_size),
    }

    SIMPLE = {
        'roulette':          lambda: roulette_wheel_selection,
        'tournament_size':   lambda: tournament_selection_min_size(pool_size, pressure_size=0.5)
                                if mode=='min'
                                else None # No max version of tournament_size
    }

    FACTORIES = {**MODED, **SIMPLE}
    if type not in FACTORIES:
        raise ValueError(f"Invalid selection type: {type}")

    return FACTORIES[type]()


def tournament_selection_min(pool_size):
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


def tournament_selection_max(pool_size):
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


def double_tournament_min(pool_size):
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


def tournament_selection_min_size(pool_size, pressure_size=1e-4):
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


def lexicase_selection(mode='min', down_sampling=0.5):
    """
    Returns a function that performs lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.
    down_sampling : float, optional
        Proportion of test cases to sample. Defaults to 0.5.

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
    def ls(population):
        """
        Perform lexicase selection on a population of individuals.
        
        Parameters
        ----------
        population : list of Individual
            The population from which to select parents.

        Returns
        -------
        Individual
            The selected parent individual.
        """
        
        # Get errors for each individual on each test case
        errors = population.errors_case
        num_cases = errors.shape[1]  
                
        # Start with all individuals in the pool
        pool = population.population.copy()
        n_cases = int(num_cases * down_sampling)  
        case_order = random.sample(range(num_cases), n_cases)
                        
        # Iterate over test cases and filter individuals based on exact performance (no epsilon)
        for i in range(n_cases):
            # Generate an int from 0 to num_cases
            case_errors = errors[:, case_order[i]]
            
            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':                
                best_individuals = np.where(case_errors == np.min(case_errors))[0]
            elif mode == 'max':
                best_individuals = np.where(case_errors == np.max(case_errors))[0]
                                                          
            # If only one individual remains, return it as the selected parent
            if len(best_individuals) == 1:
                return pool[best_individuals[0]], i+1
            
            # Filter individuals based on exact performance and error
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]
                    
        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool), i+1

    return ls  # Return the function that performs lexicase selection


def manual_epsilon_lexicase_selection(mode='min', down_sampling=0.5, epsilon=1e-6): 
    """
    Returns a function that performs manual epsilon lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.
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
    def mels(pop):
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
        Individual
            The selected parent individual.
        """
        # Get errors for each individual on each test case
        errors = pop.errors_case
        
        # Start with all individuals in the pool
        pool = pop.population.copy()
        num_cases = errors.shape[1]  # Number of test cases
        n_cases = int(num_cases * down_sampling)  # Number of test cases to sample
        case_order = random.sample(range(errors.shape[1]), n_cases)  # ADDED

        # Iterate over test cases and filter individuals based on epsilon threshold
        for i in range(n_cases):
            case_idx = case_order[i] 
            case_errors = errors[:, case_idx]  # Get errors for this test case

            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':
                best_individuals = np.where(case_errors <= np.min(case_errors) + epsilon)[0]
            elif mode == 'max':
                best_individuals = np.where(case_errors >= np.max(case_errors) - epsilon)[0]

            # If only one individual remains, return it as the selected parent
            if len(best_individuals) == 1:
                return pool[best_individuals[0]], i+1
            
            # Filter individuals based on epsilon threshold
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]

        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool), i+1
    
    return mels


def epsilon_lexicase_selection(mode='min', down_sampling=0.5):
    """
    Returns a function that performs epsilon lexicase selection to select an individual with the lowest fitness
    from a population.

    Parameters
    ----------
    mode : str, optional
        The mode of selection. Can be 'min' or 'max'. Defaults to 'min'.
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

    def els(pop):
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
        Individual
            The selected parent individual.
        """
        # Calculate the MAD for the population 
        pop.calculate_mad() 
        errors = pop.errors_case
        
        # Use the median absolute deviation to set the epsilon threshold
        num_cases = errors[0].shape[0]

        # Start with all individuals in the pool
        pool = pop.population.copy()
        # case_order = random.sample(range(num_cases), n_cases)  # ADDED

        # Iterate over test cases and filter individuals based on epsilon threshold
        n_cases = int(num_cases * down_sampling)
        for i in range(n_cases):
            # case_idx = case_order[i] 
            case_idx = random.choice(range(num_cases))
            case_errors = errors[:, case_idx]  # Get errors for this test case

            # median_case = np.median(case_errors)
            # epsilon = np.median(np.abs(case_errors - median_case))  # Compute MAD for this case
            epsilon = pop.mad[case_idx]  # Get the MAD for this case

            # Get the best error on this test case across all individuals in the pool
            if mode == 'min':
                best_individuals = np.where(case_errors <= np.min(case_errors) + epsilon)[0]
            elif mode == 'max':
                best_individuals = np.where(case_errors >= np.max(case_errors) - epsilon)[0]

            # If only one individual remains, return it as the selected parent
            if len(best_individuals) == 1:
                return pool[best_individuals[0]], i+1
            
            # Filter individuals based on epsilon threshold
            pool = [pool[i] for i in best_individuals]
            errors = errors[best_individuals]

        # If multiple individuals remain after all cases, return one at random
        return random.choice(pool), i+1

    return els


def rank_based(mode='min', pool_size=2):
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
                
        # Randomly select `pool_size` individuals for the tournament
        selected_indices = np.random.choice(len(population), pool_size, replace=False)
        
        # Find the individual with the best combined rank in the pool
        best_index = min(selected_indices, key=lambda idx: combined_ranks[idx])
        
        return population[best_index]
    
    return double_tournament


def roulette_wheel_selection(population):
    """
    Perform roulette wheel selection on a population of individuals.

    Parameters
    ----------
    population : list of Individual
        The population from which to select parents.

    Returns
    -------
    Individual
        The selected parent individual.
    """
    # return random.choices(population, weights=[ind.fitness for ind in population])[0]
    return random.choices(population)[0]
 
 
def dalex_min(down_sampling=0.5, 
              particularity_pressure=20):
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
    standardize_errors : bool, optional
        Whether to standardize the errors before performing selection. Defaults to True.
        If set to False, the errors are not standardized and the selection is performed directly on the error matrix.

    Returns
    -------
    Callable
        A function that takes a population object and returns a tuple (selected individual, n_cases used).
    """

    def ds(pop):
        # Get the error matrix (assumed shape: (n_individuals, n_total_cases))
        errors = pop.errors_case 
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

        best_index = np.argmin(F)
        return pop.population[best_index]

    return ds


def dalex_size(mode='min', 
                         down_sampling=0.5, 
                         particularity_pressure=20,
                         tournament_size=2,
                         p_best=0):
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


