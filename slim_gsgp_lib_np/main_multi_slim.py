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
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
import warnings
import numpy as np 


# -------------------------------- MULTI_SLIM --------------------------------
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.multi_slim import MULTI_SLIM
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.operators.mutators import mutator
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.operators.xo import homologus_xo
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import initializer
from slim_gsgp_lib_np.config.multi_slim_config import *

# -----------------------------------  SLIM -----------------------------------
from slim_gsgp_lib_np.utils.logger import log_settings
from slim_gsgp_lib_np.utils.utils import get_best_min, get_best_max
from slim_gsgp_lib_np.selection.selection_algorithms import selector as selection_algorithm
from slim_gsgp_lib_np.main_slim import slim
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.representations.population import Population

# -----------------------------------  GP -----------------------------------
from slim_gsgp_lib_np.utils.logger import log_settings
from slim_gsgp_lib_np.main_gp import gp

ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()
ALGORITHM = "MULTI_SLIM"

def multi_slim(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None, dataset_name: str = None,
        params_gp: dict = None, gp_version: str = "SLIM+SIG2", population: Population = None,
        pop_size : int = multi_pi_init["pop_size"], 
        n_iter : int = multi_solve_params["n_iter"],
        p_xo: float = multi_params["p_xo"],
        depth_condition: int = multi_pi_init["depth_condition"],
        max_depth: int = multi_pi_init["max_depth"],
        prob_const: float = multi_pi_init["p_c"],
        prob_terminal: float = multi_pi_init["p_t"],
        prob_specialist: float = multi_pi_init["p_specialist"],
        test_elite: bool = multi_solve_params["test_elite"],    
        n_elites: int = multi_solve_params["n_elites"],
        fitness_function : str = multi_solve_params["ffunction"],
        tournament_size: int = 2, 
        seed: int = multi_params["seed"],
        verbose: int = multi_solve_params["verbose"],
        log_level: int = multi_solve_params["log_level"],
        minimization: bool = True,
        log_path: str = None,
        selector: str = multi_params["selector"],
        std_errs: bool = multi_params["std_errs"],
        down_sampling: float = 0.5, 
        particularity_pressure: float = 20,
        dalex_size_prob: float = 0.5,
        dalex_n_cases: int = 2,
        epsilon: float = 1e-6,
        decay_rate: float = multi_params["decay_rate"],
        ensemble_functions : list = None,
        ensemble_constants : list = None,   
        callbacks: list = None, 
        timeout: int = 100,
        full_return: bool = False,
        elite_tree: Tree = None,
        it_tolerance: int = 1, 
):
    
    """
    Executes the MULTI_SLIM Genetic Programming algorithm for piecewise symbolic regression 
    with condition-based specialists.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature data.
    y_train : np.ndarray
        Training target values.
    X_test : np.ndarray, optional
        Test feature data. Default is None.
    y_test : np.ndarray, optional
        Test target values. Default is None.
    dataset_name : str, optional
        Optional name for the dataset, used in logging or result tracking.
    params_gp : dict, optional
        Dictionary of custom parameters to configure the GP algorithm.
    gp_version : str, optional
        GP variant to use (e.g., "SLIM", "SLIM+SIG", "SLIM+SIG2"). Default is "SLIM+SIG2".
    population : Population, optional
        Predefined initial population. If None, a new population is initialized.
    pop_size : int, optional
        Size of the population. Default is taken from `multi_pi_init["pop_size"]`.
    n_iter : int, optional
        Number of generations to run the evolution. Default from `multi_solve_params["n_iter"]`.
    p_xo : float, optional
        Probability of crossover. Default from `multi_params["p_xo"]`.
    depth_condition : int, optional
        Maximum depth for predicates in conditional trees.
    max_depth : int, optional
        Maximum allowed depth of any tree (entire individual).
    prob_const : float, optional
        Probability of choosing a constant when growing terminals.
    prob_terminal : float, optional
        Probability of choosing a terminal variable (vs. function).
    prob_specialist : float, optional
        Probability that a node represents a specialist (i.e., conditional form).
    test_elite : bool, optional
        If True, the best individual will be evaluated on the test set.
    n_elites : int, optional
        Number of elite individuals preserved across generations.
    fitness_function : str, optional
        Fitness function to optimize (e.g., "mse", "rmse").
    tournament_size : int, optional
        Number of individuals in tournament selection. Default is 2.
    seed : int, optional
        Random seed for reproducibility.
    verbose : int, optional
        Verbosity level (0: silent, 1: minimal, 2: detailed).
    log_level : int, optional
        Logging granularity level for tracking and callbacks.
    minimization : bool, optional
        If True, fitness is minimized (e.g., for MSE). If False, it's maximized.
    log_path : str, optional
        File path to save logs or outputs. Default is None.
    selector : str, optional
        Type of selection strategy (e.g., "tournament", "e_lexicase", "dalex", etc.).
        Default is "tournament".
    down_sampling : float, optional
        Fraction of data sampled per individual in lexicase-based selectors.
    particularity_pressure : float, optional
        Controls weight sampling in DALex selection.
    dalex_size_prob : float, optional
        Probability of selecting the individual with the best fitness in DALex size selection.
    dalex_n_cases : int, optional   
        Number of cases to sample in DALex fast selection.
    epsilon : float, optional
        Epsilon tolerance for epsilon-lexicase selection.
    decay_rate : float, optional
        Decay applied to mutation strength or error weights, depending on implementation.
    ensemble_functions : list, optional
        Functions to use for the conditional trees. Default is None, which uses the specialists functions functions.
    ensemble_constants : list, optional
        Constants to use for the conditional trees. Default is None, which uses the specialists constants.
    callbacks : list, optional
        List of callback objects to monitor or interfere with the optimization process.
    timeout : int, optional
        Maximum time allowed (in seconds) for the optimization to run.
    full_return : bool, optional
        If True, return all internal state (e.g., full population, logs); otherwise only final solution.
    elite_tree : Tree, optional
        Predefined tree to start as the elite individual. Default is None.
    it_tolerance : int, optional
        Number of iterations to wait before stopping the algorithm if no improvement is found.  

    Returns
    -------
    Best individual or (best individual, multi-slim population, specialists population) if `full_return` is True.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Validate the inputs given  TODO 
    # validate_parameters()
    if dataset_name is None:
        warnings.warn("No dataset name set. Using default value of dataset_1.")
        dataset_name = "dataset"
    
    # Calling the SLIM-GSGP algorithm 
    log_spec = None 
    if population is None:
        if type(params_gp) != dict: 
            try: 
                params_gp = params_gp.__dict__
            except: 
                print('Error: Params gp must be a dataclass or a dictionnary')

        if gp_version == "gp": 
            optimizer = gp(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name=dataset_name, test_elite=test_elite,
                full_return=True, seed=seed, verbose=verbose, log_path=log_path, 
                run_info=[ALGORITHM, gp_version, UNIQUE_RUN_ID, dataset_name], minimization=minimization,**params_gp)
        else:
            optimizer = slim(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name=dataset_name, test_elite=test_elite,
                full_return=True, seed=seed, verbose=verbose, slim_version=gp_version,
                run_info=[ALGORITHM, gp_version, UNIQUE_RUN_ID, dataset_name], minimization=minimization,
                log_path=log_path,**params_gp)
            
        population, elite = optimizer.population, optimizer.elite
        if params_gp['log_level'] == 'evaluate':
            log_spec = optimizer.log 
        population.population.sort(key=lambda x: x.fitness, reverse=not minimization)
    else: 
        elite = population[0]

    # Setting the train and test semantics for the population for speeding up evaluation during multi-slim
    for ind in population:
        ind.version = elite.version if gp_version != "gp" else None
        ind.train_semantics = ind.predict(X_train)
        ind.test_semantics = ind.predict(X_test) if X_test is not None else None
        
    # ------------------------ PI INIT ------------------------------
    if gp_version == "gp":
        multi_pi_init['FUNCTIONS'] = elite.FUNCTIONS
        multi_pi_init['TERMINALS'] = elite.TERMINALS
        multi_pi_init['CONSTANTS'] = elite.CONSTANTS

    else:
        multi_pi_init['FUNCTIONS'] = elite.collection[0].FUNCTIONS
        multi_pi_init['TERMINALS'] = elite.collection[0].TERMINALS
        multi_pi_init['CONSTANTS'] = elite.collection[0].CONSTANTS
    
    multi_pi_init['SPECIALISTS'] = {f'S_{i}' : ind for i, ind in enumerate(population.population)}

    if ensemble_functions is not None:
        multi_pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in ensemble_functions}
    if ensemble_constants is not None:
        multi_pi_init['CONSTANTS'] = {f"constant_{str(n).replace('-', '_')}": lambda _, num=n: np.array(num)
                                   for n in ensemble_constants}
        

    multi_pi_init['p_c'] = prob_const
    multi_pi_init['p_t'] = prob_terminal
    multi_pi_init['p_s'] = prob_specialist
    multi_pi_init['pop_size'] = pop_size
    multi_pi_init['depth_condition'] = depth_condition
    multi_pi_init['max_depth'] = max_depth

    # ------------------- MULTI_SLIM PARAMETERS ----------------------    
    multi_params['find_elit_func'] = get_best_min if minimization else get_best_max
    multi_params['mutator'] = mutator(FUNCTIONS=multi_pi_init['FUNCTIONS'],
                                      TERMINALS=multi_pi_init['TERMINALS'],
                                      CONSTANTS=multi_pi_init['CONSTANTS'],
                                      SPECIALISTS=multi_pi_init['SPECIALISTS'],
                                      depth_condition=multi_pi_init['depth_condition'],
                                      max_depth=multi_pi_init['max_depth'],
                                      p_c=multi_pi_init['p_c'],
                                      p_t=multi_pi_init['p_t'],
                                      decay_rate=multi_params['decay_rate'])
    
    multi_params['xo_operator'] = homologus_xo(multi_pi_init['max_depth'])
    multi_params['initializer'] = initializer
    multi_params['p_mut'] = 1 - p_xo
    multi_params['p_xo'] = p_xo
    multi_params['seed'] = seed
    multi_params['callbacks'] = callbacks
    multi_params['decay_rate'] = decay_rate
    multi_params['selector'] = selection_algorithm(problem='min' if minimization else 'max',
                                                type=selector,
                                                pool_size=tournament_size,
                                                down_sampling=down_sampling, 
                                                particularity_pressure=particularity_pressure,
                                                epsilon=epsilon,
                                                dalex_size_prob=dalex_size_prob,
                                                n_cases=dalex_n_cases,
    )
    multi_params['std_errs'] = std_errs
    
    multi_params['find_elit_func'] = get_best_min if minimization else get_best_max
    multi_params['elite_tree'] = elite_tree

    # ---------------- MULTI_SLIM SOLVE PARAMETERS --------------------
    multi_solve_params['run_info'] = [ALGORITHM, gp_version, UNIQUE_RUN_ID, dataset_name]
    multi_solve_params['ffunction'] = fitness_function_options[fitness_function]
    multi_solve_params['log_level'] = log_level
    multi_solve_params['verbose'] = verbose
    multi_solve_params['n_iter'] = n_iter
    multi_solve_params['test_elite'] = test_elite
    multi_solve_params['log_path'] = log_path
    multi_solve_params['n_elites'] = n_elites
    multi_solve_params['elitism'] = True if n_elites > 0 else False
    multi_solve_params['timeout'] = timeout
    multi_solve_params['it_tolerance'] = it_tolerance * n_iter

    # --------------- Run the MULTI_SLIM algorithm --------------------
    optimizer = MULTI_SLIM(pi_init=multi_pi_init, **multi_params)
    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **multi_solve_params
    )

    if not isinstance(log_level, str): 
        log_settings(
            path=os.path.join(os.getcwd(), "log", "slim_settings.csv"),
            settings_dict=[params_gp,
                        multi_pi_init,
                        multi_params, 
                        multi_solve_params],
            unique_run_id=UNIQUE_RUN_ID
        ) if log_level > 0 else None

    # optimizer.elite.iteration = optimizer.iteration
    # optimizer.elite.early_stop = optimizer.stop_training
    optimizer.spec_pop = population
    optimizer.spec_pop_log = log_spec
    if full_return: 
        return optimizer
    
    return optimizer.elite

