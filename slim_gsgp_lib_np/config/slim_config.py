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
import numpy as np 
from slim_gsgp_lib_np.initializers.initializers import rhh, grow, full, simple
from slim_gsgp_lib_np.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.crossovers import xo_operator
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.mutators import (deflate_mutation)
from slim_gsgp_lib_np.selection.selection_algorithms import tournament_selection_min
from slim_gsgp_lib_np.evaluators.fitness_functions import *
from slim_gsgp_lib_np.utils.utils import (get_best_min, protected_div)


# Define functions and constants
# todo use only one dictionary for the parameters of each algorithm

FUNCTIONS = {
    'add': {'function': np.add, 'arity': 2},
    'subtract': {'function': np.subtract, 'arity': 2},
    'multiply': {'function': np.multiply, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

# Generate a list of 200 constants between -1 and 1
constants = [round(-1 + (2 * i) / (100 - 1), 2) for i in range(100) if np.abs(i) > 0.1]
CONSTANTS = {f'c_{i}': lambda _: np.array(i) for i in constants}


# Set parameters
settings_dict = {"p_test": 0.2}

# SLIM GSGP solve parameters
slim_gsgp_solve_parameters = {
    "run_info": None,
    "ffunction": "rmse",
    "max_depth": 15,
    "reconstruct": True,
    "n_iter": 1000,
    "elitism": True,
    "n_elites": 1,
    "log": 1,
    "verbose": 1,
    "test_elite": True
}

# SLIM GSGP parameters
slim_gsgp_parameters = {
    "initializer": "rhh",
    "selector": 'tournament',
    "ms": None,
    "inflate_mutator": None,
    "deflate_mutator": deflate_mutation,
    "xo_operator": xo_operator,
    "p_xo": 0.0,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "p_inflate": 0.2,
    "p_struct": 0.0,
    "operator": None,
    "pop_size": 100,
    "seed": 74,
    "p_struct_xo": 0.0,
    "decay_rate": 0.1,
    "mut_xo_operator": "rshuffle",
}

slim_gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.2,
    "p_t": 0.7, 
    "init_depth": 6,
}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors
}

initializer_options = {
    "rhh": rhh,
    "grow": grow,
    "full": full,
    "simple": simple
}