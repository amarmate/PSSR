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
import math
import random
import numpy as np
from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import (create_full_random_tree,
                                                                create_grow_random_tree)
from slim_gsgp_lib_np.algorithms.GSGP.representations.tree import Tree
from sklearn.metrics import root_mean_squared_error


def protected_div(x1, x2):
    """Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return np.where(
        np.abs(x2) > 0.001,
        np.divide(x1, x2, out=np.zeros_like(x1, dtype=float), where=(x2 != 0)),
        1.0
    )

def protected_sqrt(x1):
    """Implements the square root protected against negative values

    Performs square root between x1. If x1 is (or has) negative(s), the
    function returns the absolute value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The input tensor.

    Returns
    -------
    torch.Tensor
        Result of protected square root between x1.
    """
    return np.sqrt(np.abs(x1))

def AQ_np(x, y):
    return np.divide(x, np.sqrt(1 + y**2))


def mean_(x1, x2):
    """
    Compute the mean of two tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        The first tensor.
    x2 : torch.Tensor
        The second tensor.

    Returns
    -------
    torch.Tensor
        The mean of the two tensors.
    """
    return np.divide(np.add(x1, x2), 2)



def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """Splits X and y arrays into train and test subsets.

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : np.ndarray
        Input data instances.
    y : np.ndarray
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random number generators.

    Returns
    -------
    X_train : np.ndarray
        Training data instances.
    X_test : np.ndarray
        Test data instances.
    y_train : np.ndarray
        Training target vector.
    y_test : np.ndarray
        Test target vector.
    train_indices : np.ndarray
        Indices representing the training partition.
    test_indices : np.ndarray
        Indices representing the test partition.
    """
    np.random.seed(seed)
    
    if shuffle:
        indices = np.random.permutation(X.shape[0])
    else:
        indices = np.arange(0, X.shape[0], 1)
    
    split = int(math.floor(p_test * X.shape[0]))
    
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test


def tensor_dimensioned_sum(dim):
    """
    Generate a sum function over a specified dimension.

    Parameters
    ----------
    dim : int
        The dimension to sum over.

    Returns
    -------
    function
    A function that sums tensors over the specified dimension.
    """

    def tensor_sum(input):
        return np.sum(input, dim)

    return tensor_sum

def verbose_reporter(params, 
                     first=False, 
                     precision=3,
                     col_width=20):
    """
    Prints a formatted report of custom parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing key-value pairs of parameters to be reported.
    first : bool, default=False
        Whether this is the first report to be printed.
    precision : int, default=3
        The number of decimal places to display for float values.
    col_width : int, default=20
        The width of the columns in the report.

    Returns
    -------
    None
        Outputs a formatted report to the console.
    """
    if first: 
        separator = ("+" + "-" * (col_width + 3)) * len(params) + "+"
        print(separator)
        print("".join([f"|{key.center(col_width+3)}" for key in params.keys()]) + "|")
        print(separator)

    # Print values
    values = []
    for key, value in params.items():
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif value is None:
            formatted_value = "None"
        else:
            formatted_value = str(value)
        values.append(formatted_value.center(col_width+3))
    
    print("|" + "".join([f"{value}|" for value in values]))
    print(('|' + '-' * (col_width + 3))*len(params) + '|')



def get_terminals(X):
    """
    Get terminal nodes for a dataset.

    Parameters
    ----------
    X : (torch.Tensor)
        An array to get the set of TERMINALS from, it will correspond to the columns.

    Returns
    -------
    dict
        Dictionary of terminal nodes.
    """

    return  {f"x{i}": i for i in range(len(X[0]))}


def get_best_min(population, n_elites):
    """
    Get the best individuals from the population with the minimum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        The list of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, n_elites)
        elites = [population.population[i] for i in idx[:n_elites]]
        return elites, elites[np.argmin([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmin(population.fit)]
        return [elite], elite


def get_best_max(population, n_elites):
    """
    Get the best individuals from the population with the maximum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        The list of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, -n_elites)
        elites = [population.population[i] for i in idx[-n_elites:]]
        return elites, elites[np.argmax([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmax(population.fit)]
        return [elite], elite
    

def get_random_tree(
        max_depth,
        FUNCTIONS,
        TERMINALS,
        CONSTANTS,
        inputs=None,
        p_c=0.3,
        p_t=0.5,
        p_cond=0,
        logistic=True,
):
    """
    Get a random tree using either grow or full method.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    FUNCTIONS : dict
        Dictionary of functions.
    TERMINALS : dict
        Dictionary of terminals.
    CONSTANTS : dict
        Dictionary of constants.
    inputs : torch.Tensor
        Input tensor for calculating semantics.
    p_c : float, default=0.3
        Probability of choosing a constant.
    p_t : float, default=0.5
        Probability of choosing a terminal.
    p_cond : float, default=0
        Probability of using the conditional operator.
    grow_probability : float, default=1
        Probability of using the grow method.
    logistic : bool, default=True
            Whether to use logistic semantics.

    Returns
    -------
    Tree
        The generated random tree.
    """
    tree_structure = create_grow_random_tree(
        max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=p_c, p_t=p_t, p_cond=p_cond
    )

    tree = Tree(
        structure=tree_structure,
        train_semantics=None,
        test_semantics=None,
        reconstruct=True,
    )
    if inputs is not None:
        tree.calculate_semantics(inputs, testing=False, logistic=logistic)
    return tree


def generate_random_uniform(lower, upper):
    """
    Generate a random number within a specified range using numpy random.uniform.

    Parameters
    ----------
    lower : float
        The lower bound of the range for generating the random number.
    upper : float
        The upper bound of the range for generating the random number.

    Returns
    -------
    Callable
        A function that when called, generates a random number within the specified range.
    Notes
    -----
    The returned function takes no input and returns a random float between lower and upper whenever called.
    """

    def generate_num():
        """
        Generate a random number within a specified range.

        Returns
        -------
        float
            A random number between the defined lower and upper bounds.
        """
        return random.uniform(lower, upper)

    generate_num.lower = lower
    generate_num.upper = upper
    return generate_num


def show_individual(tree, operator):
    """
    Display an individual's structure with a specified operator.

    Parameters
    ----------
    tree : Tree
        The tree representing the individual.
    operator : str
        The operator to display ('sum' or 'prod').

    Returns
    -------
    str
        The string representation of the individual's structure.
    """
    op = "+" if operator == "sum" else "*"

    return f" {op} ".join(
        [
            (
                str(t.structure)
                if isinstance(t.structure, tuple)
                else (
                    f"f({t.structure[1].structure})"
                    if len(t.structure) == 3
                    else f"f({t.structure[1].structure} - {t.structure[2].structure})"
                )
            )
            for t in tree.collection
        ]
    )


def gs_rmse(y_true, y_pred):
    """
    Calculate the root mean squared error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        The root mean squared error.
    """
    return root_mean_squared_error(y_true, y_pred[0])


def gs_size(y_true, y_pred):
    """
    Get the size of the predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    int
        The size of the predicted values.
    """
    return y_pred[1]


import numpy as np

def validate_inputs(*args, **kwargs):
    """
    Validates the inputs based on the specified conditions.

    Parameters
    ----------
    kwargs: dict
        A dictionary of parameter names and their corresponding values.
    """
    # Define the expected types for each parameter
    expected_types = {
        'X_train': np.ndarray,
        'y_train': np.ndarray,
        'X_test': (np.ndarray, type(None)),  # Can be None
        'y_test': (np.ndarray, type(None)),  # Can be None
        'pop_size': (int, np.integer),
        'n_iter': (int, np.integer),
        'elitism': bool,
        'n_elites': int,
        'init_depth': int,
        'log_path': str,
        'prob_const': (float, int),
        'tree_functions': list,
        'tree_constants': list,
        'log': (int, str), 
        'verbose': int,
        'minimization': bool,
        'test_elite': bool,
        'fitness_function': str,
        'initializer': str,
        'tournament_size': int,
        'ms_lower': int,
        'ms_upper': int,
        'p_inflate': float,
        'p_struct': float,
        'mode': str,
    }

    # Validate the type of each provided parameter
    for param, value in kwargs.items():
        if param in expected_types:
            expected_type = expected_types[param]
            if not isinstance(value, expected_type):
                raise TypeError(f"{param} must be of type {expected_type}")

    # Additional validations based on parameter values
    if 'prob_const' in kwargs and not 0 <= kwargs['prob_const'] <= 1:
        raise ValueError("prob_const must be a number between 0 and 1")

    if 'n_iter' in kwargs and kwargs['n_iter'] < 1:
        raise ValueError("n_iter must be greater than 0")

    if 'tree_constants' in kwargs:
        tree_constants = kwargs['tree_constants']
        if not all(isinstance(elem, (int, float)) and not isinstance(elem, bool) for elem in tree_constants):
            raise TypeError("tree_constants must be a list containing only integers and floats")

    if type('log') != str:
        if 'log' in kwargs and not 0 <= kwargs['log'] <= 4:
            raise ValueError("log must be between 0 and 4")

    if 'verbose' in kwargs and not 0 <= kwargs['verbose'] <= 1:
        raise ValueError("verbose must be either 0 or 1")

    if 'tournament_size' in kwargs and kwargs['tournament_size'] < 2:
        raise ValueError("tournament_size must be at least 2")

    if 'ms_lower' in kwargs and 'ms_upper' in kwargs:
        if kwargs['ms_lower'] == kwargs['ms_upper']:
            raise ValueError("ms_lower and ms_upper must be different")
        if kwargs['ms_lower'] > kwargs['ms_upper']:
            raise ValueError("ms_lower must be smaller than ms_upper")

    if 'p_inflate' in kwargs and kwargs['p_inflate'] < 0:
        raise ValueError("p_inflate must be greater or equal to 0")

    if 'p_struct' in kwargs and kwargs['p_struct'] < 0:
        raise ValueError("p_struct must be greater or equal to 0")

    if 'p_inflate' in kwargs and 'p_struct' in kwargs:
        if kwargs['p_inflate'] + kwargs['p_struct'] > 1:
            raise ValueError("p_inflate + p_struct must be smaller or equal to 1")

    if 'mode' in kwargs and kwargs['mode'] not in ["normal", "exp", "uniform"]:
        raise ValueError("mode must be one of: 'normal', 'exp', 'uniform'")

    # Ensure that if test_elite is True, X_test and y_test must not be None
    if kwargs.get('test_elite', False):
        if kwargs.get('X_test') is None or kwargs.get('y_test') is None:
            raise ValueError("If test_elite is True, X_test and y_test cannot be None")

    # Add more validations as necessary based on the parameters provided
    

def check_slim_version(slim_version):
    """
    Validate the slim_gsgp version given as input bu the users and assign the correct values to the parameters op, sig and trees
    Parameters
    ----------
    slim_version : str
        Name of the slim_gsgp version.

    Returns
    -------
    op, sig, trees
        Parameters reflecting the kind of operation considered, the use of the sigmoid and the use of multiple trees.
    """
    if slim_version == "SLIM+SIG2":
        return "sum", True, True
    elif slim_version == "SLIM*SIG2":
        return "mul", True, True
    elif slim_version == "SLIM+ABS":
        return "sum", False, False
    elif slim_version == "SLIM*ABS":
        return "mul", False, False
    elif slim_version == "SLIM+SIG1":
        return "sum", True, False
    elif slim_version == "SLIM*SIG1":
        return "mul", True, False
    else:
        raise Exception('Invalid SLIM configuration')

def _evaluate_slim_individual(individual, ffunction, y, testing=False, operator="sum"):
    """
    Evaluate the individual using a fitness function.

    Args:
        ffunction: Fitness function to evaluate the individual.
        y: Expected output (target) values as a torch tensor.
        testing: Boolean indicating if the evaluation is for testing semantics.
        operator: Operator to apply to the semantics ("sum" or "prod").

    Returns:
        None
    """
    if operator == "sum":
        operator = np.sum
    else:
        operator = np.prod

    if testing:
        individual.test_fitness = ffunction(
            y,
            np.clip(
                operator(individual.test_semantics, dim=0),
                -1000000000000.0,
                1000000000000.0,
            ),
        )

    else:
        individual.fitness = ffunction(
            y,
            np.clip(
                operator(individual.train_semantics, dim=0),
                -1000000000000.0,
                1000000000000.0,
            ),
        )

        # if testing is false, return the value so that training parallelization has effect
        return ffunction(
                y,
                np.clip(
                    operator(individual.train_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )
    

def get_indices(tree, path=()):
    """
    Get all indices that can be used to access valid subtrees or terminal nodes in a tree.

    Parameters
    ----------
    tree : tuple
        The current node of the tree.
    path : tuple
        The path to the current node.

    Returns
    -------
    list
        A list of all indices that can be used to access valid elements in the tree.
    """
    indices = []
    
    # If not tuple 
    if not isinstance(tree, tuple):
        indices.append(path)

    # If tuple, separate
    else:
        indices.append(path) if path != () else None
        op, left, right = tree  
        # Can substitute the left or right
        indices.extend(get_indices(left, path + (1,)))
        indices.extend(get_indices(right, path + (2,)))

    return indices

def get_indices_with_levels(tree):
    """
    Get all index‐paths into `tree` and their depths. Works for tuples
    of the form (op, child1, child2, ..., childN), for any N>=0.

    Returns
    -------
    List[(path: tuple[int], level: int)]
    """
    def traverse(sub_tree, path=(), level=0):
        nodes = [(path, level)]
        # only recurse into tuple‐nodes with children
        if isinstance(sub_tree, tuple) and len(sub_tree) > 1:
            # children are at positions 1..end
            for child_idx in range(1, len(sub_tree)):
                child = sub_tree[child_idx]
                nodes.extend(traverse(child,
                                      path + (child_idx,),
                                      level + 1))
        return nodes

    return traverse(tree)


def swap_sub_tree(tree, new_tree, indices):
    """
    Swap a subtree in a tree.

    Parameters
    ----------
    tree : tuple
        The current node of the tree.
    new_tree : tuple        
        The new subtree to be swapped.
    indices : list
        The indices of the tree to be swapped.
    """
    if indices == []:
        return new_tree
    
    index = indices[0]
    return tree[:index] + (swap_sub_tree(tree[index],new_tree, indices[1:]),) + tree[index+1:]


def get_subtree(tree, indices):
        """
        Get a subtree in a tree.

        Parameters
        ----------
        tree : tuple
            The current node of the tree.
        indices : list
            The indices of the tree to be accessed.

        Returns
        -------
        tuple
            The subtree in the tree.
        """
        if indices == []:
            return tree

        index = indices[0]
        return get_subtree(tree[index], indices[1:])



