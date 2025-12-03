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
This script sets up the configuration dictionaries for the execution of the GP algorithm
"""
from slim_gsgp_lib_np.initializers.initializers import rhh, grow, full
from slim_gsgp_lib_np.evaluators.fitness_functions import *
from slim_gsgp_lib_np.utils.utils import protected_div, protected_sqrt, AQ_np
import numpy as np


FUNCTIONS = {
    'add': {'function': np.add, 'arity': 2},
    'subtract': {'function': np.subtract, 'arity': 2},
    'multiply': {'function': np.multiply, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2},
    'sqrt' : {'function': protected_sqrt, 'arity': 1},
    'cond' : {'function': lambda x, y, z: np.where(x > 0, y, z), 'arity': 3},
    'sq' : {'function': np.square, 'arity': 1},
    'AQ' : {'function': AQ_np, 'arity': 2},
}

# CONSTANTS = {
#     'constant_2': lambda _: np.array(2.0),
#     'constant_3': lambda _: np.array(3.0),
#     'constant_4': lambda _: np.array(4.0),
#     'constant_5': lambda _: np.array(5.0),
#     'constant__1': lambda _: np.array(-1.0)
# }

# constants = [round(-1 + (2 * i) / (40 - 1), 2) for i in range(40) if np.abs(i) > 0.1]
# CONSTANTS = {f'constant_{i}': lambda _: np.array(i) for i in constants}

constants = [round(i*0.05, 2) for i in range(2, 21)] + [round(-i*0.05, 2) for i in range(2, 21)]
CONSTANTS = {f'constant_{i}': lambda _: np.array(i) for i in constants}

functions = ['add', 'subtract', 'multiply', 'divide', 'sqrt', 'sq']


# Set parameters
settings_dict = {"p_test": 0.2}

# GP solve parameters
gp_solve_parameters = {
    "log_level": 1,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": "rmse",
    "n_jobs": 1,
    "max_depth": 17,
    "n_elites": 1,
    "elitism": True,
    "n_iter": 1000
}

# GP parameters
gp_parameters = {
    "initializer": "rhh",
    "selector": 'tournament',
    "std_errs": True,
    "settings_dict": settings_dict,
    "p_xo": 0.8,
    "pop_size": 100,
    "seed": 74
}

gp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.2,
    "p_t": 0.7,
    "p_cond" : 0, 
    "init_depth": 6
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
    "full": full
}