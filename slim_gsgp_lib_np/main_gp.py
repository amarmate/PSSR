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
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
import warnings
from slim_gsgp_lib_np.algorithms.GP.gp import GP
from slim_gsgp_lib_np.algorithms.GP.operators.mutators import mutator
from slim_gsgp_lib_np.algorithms.GP.operators.crossover_operators import crossover_trees
from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import tree_depth
from slim_gsgp_lib_np.config.gp_config import *
from slim_gsgp_lib_np.selection.selection_algorithms import selector as selection_algorithm
from slim_gsgp_lib_np.utils.logger import log_settings
from slim_gsgp_lib_np.utils.utils import (get_terminals, validate_inputs, get_best_max, get_best_min)
import numpy as np 
import random 

def gp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None,
       dataset_name: str = None,
       pop_size: int = gp_parameters["pop_size"],
       n_iter: int = gp_solve_parameters["n_iter"],
       p_xo: float = gp_parameters['p_xo'],
       elitism: bool = gp_solve_parameters["elitism"], n_elites: int = gp_solve_parameters["n_elites"],
       selector: str = gp_parameters["selector"],
       std_errs: bool = gp_parameters["std_errs"],
       dalex_size_prob: float = 0.5,
       dalex_n_cases: int = 2, 
       max_depth: int | None = gp_solve_parameters["max_depth"],
       init_depth: int = gp_pi_init["init_depth"],
       log_path: str = None, seed: int = gp_parameters["seed"],
       log_level: int = gp_solve_parameters["log_level"],
       verbose: int = gp_solve_parameters["verbose"],
       minimization: bool = True,
       fitness_function: str = gp_solve_parameters["ffunction"],
       initializer: str = gp_parameters["initializer"],
       n_jobs: int = gp_solve_parameters["n_jobs"],
       prob_const: float = gp_pi_init["p_c"],
       prob_terminal: float = gp_pi_init["p_t"],
       prob_cond: float = gp_pi_init["p_cond"],
       tree_functions: list = functions,
       tree_constants: list = [float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
       tournament_size: int = 2,
       down_sampling: float = 1, 
       particularity_pressure: float = 20,
       epsilon: float = 1e-6,
       test_elite: bool = gp_solve_parameters["test_elite"],
       run_info: list = None,
       callbacks: list = None,
       full_return: bool = False,
       elite_tree: list = None,
       it_tolerance: int = 1, 
       ):

    """
    Main function to execute the StandardGP algorithm on specified datasets

    Parameters
    ----------
    X_train: (np.ndarray)
        Training input data.
    y_train: (np.ndarray)
        Training output data.
    X_test: (np.ndarray), optional
        Testing input data.
    y_test: (np.ndarray), optional
        Testing output data.
    dataset_name : str, optional
        Dataset name, for logging purposes
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    selector : str, optional
        The selection algorithm to use. Options include "tournament", "lexicase", "e_lexicase", 
        "manual_e_lexicase", "dalex", "dalex_size", "rank_based", "double_tournament"
    dalex_size_prob : float, optional   
        Probability of selecting the individual with the best fitness from the pool in the dalex_size selection algorithm (default is 0.5).
    dalex_n_cases : int, optional   
        Number of cases to use in the dalex selection fast algorithm (default is 2).
    n_elites : int, optional
        The number of elites.
    max_depth : int, optional
        The maximum depth for the GP trees.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved. Defaults to `os.path.join(os.getcwd(), "log", "gp.csv")`
    seed : int, optional
        Seed for the randomness
    log_level : int, optional
        Level of detail to utilize in logging. If set to 'evaluate', the algorithm will return the log together with the elite.
    verbose : int, optional
       Level of detail to include in console output.
    minimization : bool, optional
        If True, the objective is to minimize the fitness function. If False, maximize it (default is True).
    fitness_function : str, optional
        The fitness function used for evaluating individuals (default is from gp_solve_parameters).
    initializer : str, optional
        The strategy for initializing the population (e.g., "grow", "full", "rhh").
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    prob_const : float, optional
        The probability of a constant being chosen rather than a terminal in trees creation (default: 0.2).
    prob_terminal : float, optional
        The probability of a terminal being chosen rather than a function in trees creation (default: 0.7).
    prob_cond : float, optional
        The probability of a conditional function being chosen rather than a function in trees creation (default: 0.0).
    tree_functions : list, optional
        List of allowed functions that can appear in the trees. Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    tournament_size : int, optional
        Tournament size to utilize during selection. Only applicable if using tournament selection. (Default is 2)
    down_sampling : float, optional
        Down sampling value to use for the particularity selection algorithm (default is 0.5).
    particularity_pressure : float, optional
        Pressure to apply to the particularity selection algorithm (default is 20).
    epsilon : float, optional
        Epsilon value to use for manual epsilon lexicase selection (default is 1e-6).
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.
    run_info : list, optional
        Information about the run (default is None).
    callbacks : list, optional
        List of callbacks to use during the optimization process.
    full_return : bool, optional
        If True, returns the elite and full population. If False, returns only the best individual.
    elite_tree : List of trees, optional
        Elite trees to add to the original population.

    Returns
    -------
    Tree
        Returns the best individual at the last generation.
    """

    # ================================
    #         Input Validation
    # ================================

    random.seed(seed)
    np.random.seed(seed)

    # Setting the log_path
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "log", "gp.csv")

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter,
                    elitism=elitism, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const,
                    tree_functions=tree_functions, tree_constants=tree_constants, log=log_level, verbose=verbose,
                    minimization=minimization, n_jobs=n_jobs, test_elite=test_elite, fitness_function=fitness_function,
                    initializer=initializer, tournament_size=tournament_size)


    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    if not (isinstance(max_depth, int) or isinstance(max_depth, np.integer)) and max_depth is not None:
        raise TypeError("max_depth value must be a int or None")

    assert max_depth is None or init_depth <= max_depth, f"max_depth must be at least {init_depth}"

    if dataset_name is None:
        warnings.warn("No dataset name set. Using default value of dataset_1.")
        dataset_name = "dataset_1"


    # creating a list with the valid available fitness functions
    valid_fitnesses = list(fitness_function_options)

    # assuring the chosen fitness_function is valid
    assert fitness_function.lower() in fitness_function_options.keys(), \
        "fitness function must be: " + f"{', '.join(valid_fitnesses[:-1])} or {valid_fitnesses[-1]}" \
            if len(valid_fitnesses) > 1 else valid_fitnesses[0]


    # creating a list with the valid available initializers
    valid_initializers = list(initializer_options)

    # assuring the chosen initializer is valid
    assert initializer.lower() in initializer_options.keys(), \
        "initializer must be " + f"{', '.join(valid_initializers[:-1])} or {valid_initializers[-1]}" \
            if len(valid_initializers) > 1 else valid_initializers[0]
    
    if log_level == 'evaluate' and not test_elite:
        warnings.warn("If log_level is set to 'evaluate', test_elite must be set to True. Setting log_level to 0")
        log_level = 0

    if 'cond' in tree_functions and prob_cond > 0: 
        warnings.warn("p_cond can't be larger than 0 if 'cond' not in FUNCTIONS. Setting p_cond to 0")
        prob_cond = 0

    # ================================
    #       Parameter Definition
    # ================================

    if not elitism:
        n_elites = 0

    unique_run_id = uuid.uuid1()

    algo = "StandardGP"

    #   *************** GP_PI_INIT ***************
    TERMINALS = get_terminals(X_train)
    gp_pi_init["TERMINALS"] = TERMINALS
    try:
        gp_pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in tree_functions}
    except KeyError as e:
        valid_functions = list(FUNCTIONS)
        raise KeyError(
            "The available tree functions are: " + f"{', '.join(valid_functions[:-1])} or "f"{valid_functions[-1]}"
            if len(valid_functions) > 1 else valid_functions[0])

    try:
        gp_pi_init['CONSTANTS'] = {f"constant_{str(n).replace('-', '_')}": lambda _, num=n: np.array(num)
                                   for n in tree_constants}
    except KeyError as e:
        valid_constants = list(CONSTANTS)
        raise KeyError(
            "The available tree constants are: " + f"{', '.join(valid_constants[:-1])} or "f"{valid_constants[-1]}"
            if len(valid_constants) > 1 else valid_constants[0])
        
    gp_pi_init["p_c"] = prob_const
    gp_pi_init["p_t"] = prob_terminal
    gp_pi_init["p_cond"] = prob_cond
    gp_pi_init["init_pop_size"] = pop_size # TODO: why init pop_size != than rest?
    gp_pi_init["init_depth"] = init_depth

    #  *************** GP_PARAMETERS ***************
    gp_parameters["p_xo"] = p_xo
    gp_parameters["p_m"] = 1 - gp_parameters["p_xo"]
    gp_parameters["pop_size"] = pop_size
    gp_parameters["mutator"] = mutator(
        FUNCTIONS=gp_pi_init['FUNCTIONS'],
        TERMINALS=gp_pi_init['TERMINALS'],
        CONSTANTS=gp_pi_init['CONSTANTS'],
        max_depth=max_depth,
        p_c=gp_pi_init['p_c'],
        p_t=gp_pi_init['p_t'],
        p_cond=gp_pi_init['p_cond'],
    )

    gp_parameters["crossover"] = crossover_trees(FUNCTIONS=FUNCTIONS, max_depth=max_depth)

    gp_parameters["initializer"] = initializer_options[initializer]

    gp_parameters["selector"] = selection_algorithm(problem='min' if minimization else 'max', 
                                         type=selector, 
                                         pool_size=tournament_size, 
                                         down_sampling=down_sampling,
                                         particularity_pressure=particularity_pressure, 
                                         epsilon=epsilon, 
                                         dalex_size_prob=dalex_size_prob,
                                         n_cases=dalex_n_cases,
                                         n=X_train.shape[0],
    )
    gp_parameters["std_errs"] = std_errs

    gp_parameters["find_elit_func"] = get_best_min if minimization else get_best_max
    gp_parameters["seed"] = seed
    gp_parameters["callbacks"] = callbacks

    if type(elite_tree) != list:
        elite_tree = [elite_tree]
    
    if elite_tree[0] is None:
        elite_tree = None 
    
    gp_parameters["elite_tree"] = elite_tree
    
    #   *************** GP_SOLVE_PARAMETERS ***************

    gp_solve_parameters['run_info'] = [algo, unique_run_id, dataset_name] if run_info is None else run_info
    gp_solve_parameters["log_level"] = log_level
    gp_solve_parameters["verbose"] = verbose
    gp_solve_parameters["log_path"] = log_path
    gp_solve_parameters["elitism"] = elitism
    gp_solve_parameters["n_elites"] = n_elites
    gp_solve_parameters["max_depth"] = max_depth
    gp_solve_parameters["n_iter"] = n_iter
    gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=gp_pi_init['FUNCTIONS'])
    gp_solve_parameters["ffunction"] = fitness_function_options[fitness_function]
    gp_solve_parameters["n_jobs"] = n_jobs
    gp_solve_parameters["test_elite"] = test_elite
    gp_solve_parameters['it_tolerance'] = it_tolerance * n_iter

    # ================================
    #       Running the Algorithm
    # ================================

    optimizer = GP(pi_init=gp_pi_init, **gp_parameters)
    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **gp_solve_parameters
    )

    log_settings(
        path=log_path[:-4] + "_settings.csv",
        settings_dict=[gp_solve_parameters,
                       gp_parameters,
                       gp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )

    if full_return: 
        return optimizer
    return optimizer.elite


# if __name__ == "__main__":
#     from slim_gsgp_lib_np.datasets.data_loader import load_resid_build_sale_price
#     from slim_gsgp_lib_np.utils.utils import train_test_split

#     X, y = load_resid_build_sale_price(X_y=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

#     final_tree = gp(X_train=X_train, y_train=y_train,
#                     X_test=X_val, y_test=y_val,
#                     dataset_name='resid_build_sale_price', pop_size=100, n_iter=1000, prob_const=0, fitness_function="rmse", n_jobs=2)

#     final_tree.print_tree_representation()
#     predictions = final_tree.predict(X_test)
#     print(float(rmse(y_true=y_test, y_pred=predictions)))

if __name__ == "__main__":
    from slim_gsgp_lib_np.datasets.data_loader import load_airfoil
    from slim_gsgp_lib_np.utils.utils import train_test_split

    X, y = load_airfoil(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2)

    final_tree = gp(X_train=X_train, y_train=y_train, max_depth=7, selector='dalex', particularity_pressure=10,
                    dataset_name='airfoil', pop_size=100, n_iter=1000, test_elite=False)