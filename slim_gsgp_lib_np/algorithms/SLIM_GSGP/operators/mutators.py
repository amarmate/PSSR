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
Mutation Functions for SLIM GSGP.
"""

import random
import numpy as np
from slim_gsgp_lib_np.algorithms.GSGP.representations.tree import Tree
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp_lib_np.utils.utils import get_random_tree, swap_sub_tree, get_indices, get_indices_with_levels
from functools import lru_cache

# two tree function
def two_trees_delta(operator="sum"):
    """
    Generate a function for the two-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).

    Returns
    -------
    Callable
        A mutation function (`tt_delta`) for two Individuals that returns the mutated semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree individual.
        tr2 : Individual
            The second tree individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        np.Tensor
            The mutated semantics.

    Notes
    -----
    The returned function ('tt_delta_{operator}') takes as input two individuals, the mutation step, a boolean
    representing whether to use the train or test semantics, and returns the calculated semantics of the new individual.
    """

    def tt_delta(tr1, tr2, ms, testing):
        """
        Performs delta mutation between two trees based on their semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree Individual.
        tr2 : Individual
            The second tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        np.array
            The mutated semantics.
        """
        if testing:
            # Test semantics
            if tr1.test_semantics is None or tr2.test_semantics is None:
                raise ValueError("Semantics not calculated for tr1 or tr2")
            
            return (
                np.multiply(ms, np.subtract(tr1.test_semantics, tr2.test_semantics))
                if operator == "sum"
                else np.add(
                    1, np.multiply(ms, np.subtract(tr1.test_semantics, tr2.test_semantics))
                )
            )

        else:
            return (
                np.multiply(ms, np.subtract(tr1.train_semantics, tr2.train_semantics))
                if operator == "sum"
                else np.add(
                    1,
                    np.multiply(ms, np.subtract(tr1.train_semantics, tr2.train_semantics)),
                )
            )

    tt_delta.__name__ += "_" + operator

    return tt_delta


def one_tree_delta(operator="sum", sig=False):
    """
    Generate a function for the one-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        A mutation function (`ot_delta`) for one-tree mutation.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        np.Tensor
            The mutated semantics.
    Notes
    -----
    The returned function ('ot_delta_{operator}_{sig}') takes as input one individual, the mutation step,
    a boolean representing whether to use the train or test semantics, and returns the mutated semantics.
    """
    def ot_delta(tr1, ms, testing):
        """
        Performs delta mutation on one tree based on its semantics.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        np.Tensor
            The mutated semantics.
        """
        # Check if tr1 has semantics
        if sig:
            if testing:
                if tr1.test_semantics is None:
                    raise ValueError("Semantics not calculated for tr1")
                return (
                    np.multiply(ms, np.subtract(np.multiply(2, tr1.test_semantics), 1))
                    if operator == "sum"
                    else np.add(
                        1, np.multiply(ms, np.subtract(np.multiply(2, tr1.test_semantics), 1))
                    )
                )
            else:
                return (
                    np.multiply(ms, np.subtract(np.multiply(2, tr1.train_semantics), 1))
                    if operator == "sum"
                    else np.add(
                        1,
                        np.multiply(ms, np.subtract(np.multiply(2, tr1.train_semantics), 1)),
                    )
                )
        else:
            if testing:
                if tr1.test_semantics is None:
                    raise ValueError("Semantics not calculated for tr1")

                return (
                    np.multiply(
                        ms,
                        np.subtract(
                            1, np.divide(2, np.add(1, np.abs(tr1.test_semantics)))
                        ),
                    )
                    if operator == "sum"
                    else np.add(
                        1,
                        np.multiply(
                            ms,
                            np.subtract(
                                1,
                                np.divide(
                                    2, np.add(1, np.abs(tr1.test_semantics))
                                ),
                            ),
                        ),
                    )
                )
            else:
                return (
                    np.multiply(
                        ms,
                        np.subtract(
                            1,
                            np.divide(2, np.add(1, np.abs(tr1.train_semantics))),
                        ),
                    )
                    if operator == "sum"
                    else np.add(
                        1,
                        np.multiply(
                            ms,
                            np.subtract(
                                1,
                                np.divide(
                                    2, np.add(1, np.abs(tr1.train_semantics))
                                ),
                            ),
                        ),
                    )
                )

    ot_delta.__name__ += "_" + operator + "_" + str(sig)
    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS, two_trees=True, operator="sum", single_tree_sigmoid=False, sig=False):
    """
    Generate an inflate mutation function.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    two_trees : bool
        Boolean indicating if two trees should be used.
    operator : str
        The operator to be used in the mutation.
    single_tree_sigmoid : bool
        Boolean indicating if sigmoid should be applied to a single tree.
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        An inflate mutation function (`inflate`).

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : np.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : np.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.

    Notes
    -----
    The returned function performs inflate mutation on Individuals, using either one or two randomly generated trees
    and applying either delta mutation or sigmoid mutation based on the parameters.
    """
    
    def inflate(
        individual,
        ms,
        X,
        max_depth=8,
        p_c=0.1,
        p_t=0.5,    
        X_test=None,
        reconstruct=True,
        ):
        """
        Perform inflate mutation on the given Individual.

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : np.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        p_t : float, optional
            Probability of terminal selection (default: 0.5).
        X_test : np.Tensor, optional
            Test data for calculating test semantics (default: None).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.
        """
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                p_t=p_t,
                logistic=True,
            )
            random_tree2 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                p_t=p_t,
                logistic=True,
            )
            
            # adding the random trees to a list, to be used in the creation of a new block
            random_trees = [random_tree1, random_tree2]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(X_test, testing=True, logistic=True)
                    for rt in random_trees
                ]

        if not two_trees:
            # getting one random tree
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                p_t=p_t,
                logistic=single_tree_sigmoid or sig,
            )
            # adding the random tree to a list, to be used in the creation of a new block
            random_trees = [random_tree1]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(
                        X_test, testing=True, logistic=single_tree_sigmoid or sig
                    )
                    for rt in random_trees
                ]

        # getting the correct mutation operator, based on the number of random trees used
        variator = (
            two_trees_delta(operator=operator)
            if two_trees
            else one_tree_delta(operator=operator, sig=sig)
        )        

        # creating the new block for the individual, based on the random trees and operators
        new_block = Tree(
            structure=[variator, *random_trees, ms],
            train_semantics=variator(*random_trees, ms, testing=False),
            test_semantics=(
                variator(*random_trees, ms, testing=True)
                if X_test is not None
                else None
            ),
            reconstruct=True,
        )
        # creating the offspring individual, by adding the new block to it
        offs = Individual(
            collection=[*individual.collection, new_block] if reconstruct else None,
            train_semantics=np.stack(
                [
                    *individual.train_semantics,
                    (
                        new_block.train_semantics
                        if new_block.train_semantics.shape != ()
                        else new_block.train_semantics.repeat(len(X))
                    ),
                ]
            ),
            test_semantics=(
                (
                    np.stack(
                        [
                            *individual.test_semantics,
                            (
                                new_block.test_semantics
                                if new_block.test_semantics.shape != ()
                                else new_block.test_semantics.repeat(len(X_test))
                            ),
                        ]
                    )
                )
                # if individual.test_semantics is not None
                if False
                else None
            ),
            reconstruct=reconstruct,
        )
        # computing offspring attributes
        offs.size = individual.size + 1
        offs.nodes_collection = [*individual.nodes_collection, new_block.nodes]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [*individual.depth_collection, new_block.depth]
        offs.depth = max(
            [
                depth - (i - 1) if i != 0 else depth
                for i, depth in enumerate(offs.depth_collection)
            ]
        ) + (offs.size - 1)
        
        offs.age = individual.age + 1
        offs.id = individual.id

        return offs

    return inflate

def deflate_mutation(individual, reconstruct, mut_point_idx=None):
    """
    Perform deflate mutation on a given Individual by removing a random 'block'.

    Parameters
    ----------
    individual : Individual
        The Individual to be mutated.
    reconstruct : bool
        Whether to store the Individual's structure after mutation.
    mut_point_idx : int, optional
        The index of the block to be removed (default: None).

    Returns
    -------
    Individual
        The mutated individual
    """
    # choosing the block that will be removed
    try:
        mut_point = random.randint(1, individual.size - 1) if mut_point_idx is None else mut_point_idx
    except:
        print("Error: ", individual.size, mut_point_idx)
        print("Individual: ", individual.structure)
        raise ValueError("Error: ", individual.size, mut_point_idx)

    # removing the block from the individual and creating a new Individual
    offs = Individual(
        collection=(
            [
                *individual.collection[:mut_point],
                *individual.collection[mut_point + 1 :],
            ]
            if reconstruct
            else None
        ),
        train_semantics=np.stack(
            [
                *individual.train_semantics[:mut_point],
                *individual.train_semantics[mut_point + 1 :],
            ]
        ),
        test_semantics=(
            np.stack(
                [
                    *individual.test_semantics[:mut_point],
                    *individual.test_semantics[mut_point + 1 :],
                ]
            )
            # if individual.test_semantics is not None
            if False
            else None
        ),
        reconstruct=reconstruct,
    )

    # computing offspring attributes
    offs.size = individual.size - 1
    offs.nodes_collection = [
        *individual.nodes_collection[:mut_point],
        *individual.nodes_collection[mut_point + 1 :],
    ]
    offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

    offs.depth_collection = [
        *individual.depth_collection[:mut_point],
        *individual.depth_collection[mut_point + 1 :],
    ]
    offs.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs.depth_collection)
        ]
    ) + (offs.size - 1)
    
    offs.age = individual.age + 1 if mut_point_idx is None else individual.age
    offs.id = individual.id

    return offs


# ----------------------------- ADDED ----------------------------- #
@lru_cache(maxsize=128)
def exp_decay_prob(n, decay_rate=0.1, invert=False):
    """
    Generate an exponential decay probability distribution.
    
    Parameters
    ----------
    n : int
        Number of elements in the distribution.
    decay_rate : float, optional
        Decay rate for the exponential distribution (default: 0.1).
    invert : bool, optional
        Flag to indicate whether the distribution should be inverted (default: False).

    Returns
    -------
    np.ndarray
        The exponential decay probability distribution.
    """

    prob = np.exp(-decay_rate * np.arange(n))
    prob = prob[::-1] if invert else prob
    return prob / np.sum(prob)

def exp(individual_tree_depth, 
        max_depth, 
        indices_with_levels,
        decay_rate):
    """
    Helps sturcutre mutation choose a mutation tree depth based on an exponential distribution.

    Parameters
    ----------
    individual_tree_depth : int
        Depth of the individual tree.
    max_depth : int
        Maximum depth for generated trees.
    indices_with_levels : list
        List of indices with their levels.
    decay_rate : float
        Decay rate for the exponential distribution.

    Returns
    -------
    int
        Random index for the mutation.
    int
        Depth for the mutation.
    """
    mut_level = random.choice([i for i in range(0, individual_tree_depth)])

    if mut_level == 0:
        # The root node is selected, so we can only insert trees (not nodes)
        random_index = indices_with_levels[0][0]
        prob_decay = exp_decay_prob(max_depth-1, decay_rate=decay_rate, invert=False)
        depth = np.random.choice(np.arange(2, max_depth+1), p=prob_decay)  # np.arange(2,4)=[2,3] only
    else:
        random_index = random.choice([i for i, level in indices_with_levels if level == mut_level])
        prob_decay = exp_decay_prob(max_depth-mut_level, decay_rate=decay_rate, invert=False)
        depth = np.random.choice(np.arange(1, max_depth-mut_level+1), p=prob_decay)
    return random_index, depth


def uniform(max_depth, 
            indices_with_levels,
            *args):
    """
    Helps sturcutre mutation choose a mutation tree depth based on a uniform distribution over each of the possible indices.

    Parameters
    ----------
    max_depth : int
        Maximum depth for generated trees.
    indices_with_levels : list
        List of indices with their levels.

    Returns
    -------
    int
        Random index for the mutation.
    int
        Depth for the mutation.
    """
    random_index = random.choice([(key,level) for key, level in indices_with_levels])
    if random_index[1] == 0:
        depth = random.choice(np.arange(2, max_depth))
    else:
        depth = random.choice(np.arange(1, max_depth-random_index[1]+1))

    return random_index[0], depth

def normal(individual_tree_depth,
            max_depth,
            indices_with_levels,
            decay_rate,
            ):
    """
    Helps sturcutre mutation choose a mutation tree depth based on a normal distribution over each of the possible indices.

    Parameters
    ----------
    individual_tree_depth : int
        Depth of the individual tree.
    max_depth : int
        Maximum depth for generated trees.
    indices_with_levels : list
        List of indices with their levels.
    decay_rate : float
        Decay rate for the exponential distribution.

    Returns
    -------
    int
        Random index for the mutation.
    int
        Depth for the mutation.
    """

    random_index = random.choice([(key,level) for key, level in indices_with_levels])
    
    # IMPLEMENTATION 
    pass


depth_distribution_functions = {
        "exp": lambda individual_tree_depth, max_depth, indices_with_levels, decay_rate: exp(individual_tree_depth, max_depth, indices_with_levels, decay_rate),
        "uniform": lambda individual_tree_depth, max_depth, indices_with_levels, decay_rate: uniform(max_depth, indices_with_levels, individual_tree_depth, decay_rate),
        "normal": lambda individual_tree_depth, max_depth, indices_with_levels, decay_rate: normal(individual_tree_depth, max_depth, indices_with_levels, decay_rate),
}


def structure_mutation(FUNCTIONS, TERMINALS, CONSTANTS, mode="exp"):
    """
    Generate a function for the structure mutation.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    mode : str, optional
        The mode of the mutation (default: "exp"). Choose between "exp", "uniform" and "normal".
    Returns
    -------
    Callable
        A structure mutation function (`structure`).

    Notes 
    -------
    Until now, no function mutation has been implemented, so when selecting a node, it is always replaced by a new tree or terminal (pruning).
    """
    
    def structure(individual,
                        X,
                        max_depth=8,
                        p_c=0.1,
                        p_t=0.5,
                        X_test=None,
                        reconstruct=True, 
                        decay_rate=0.2,
                        **args,
    ):
        """
        Perform a mutation on a given Individual by changing the main structure of the tree.

        Parameters
        ----------
        individual : Individual
            The Individual to be mutated.
        X : np.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        p_t : float, optional
            Probability of terminal selection (default: 0.5).
        X_test : np.Tensor, optional
            Test data for calculating test semantics (default: None).
        X_test : np.Tensor, optional
            Test data for calculating test semantics (default: None).
        reconstruct : bool
            Whether to store the Individuals structure after mutation.
        decay_rate : float, optional
            Decay rate for the exponential distribution (default: 0.2).

        Returns
        -------
        Individual
            The mutated individual
        """
        individual_tree_depth = individual.collection[0].depth
        indices_with_levels = get_indices_with_levels(individual.structure[0])
        random_index, depth = depth_distribution_functions[mode](individual_tree_depth, max_depth, indices_with_levels, decay_rate)
        
        # ---------------------------------------------------------------------------------------------------------------
        # If just a node is selected
        if depth == 1:
            if random.random() < p_c:
                new_block = random.choice(list(CONSTANTS.keys()))
            else:
                new_block = random.choice(list(TERMINALS.keys()))
            
            # Swap the subtree in the main tree
            new_structure = swap_sub_tree(individual.structure[0], new_block, list(random_index))

        # Else generate a tree        
        else:
            rt = get_random_tree(
            depth,
            FUNCTIONS,
            TERMINALS,
            CONSTANTS,
            inputs=X,
            p_c=p_c,
            p_t=p_t,
            logistic=False,
        )         

            # Swap the subtree in the main tree
            new_structure = swap_sub_tree(individual.structure[0], rt.structure, list(random_index))
    
        # Create the new block
        new_block = Tree(structure=new_structure,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True)
        
        new_block.calculate_semantics(X)            
        
        # Create the offspring individual
        if X_test is not None:
            new_block.calculate_semantics(X_test, testing=True, logistic=False)
            
        offs = Individual(
            collection=[new_block, *individual.collection[1:]],
            train_semantics=np.stack(
                [
                    new_block.train_semantics,
                    *individual.train_semantics[1:],
                ]   
            ),
            test_semantics=(
                np.stack(
                    [
                        new_block.test_semantics,
                        *individual.test_semantics[1:],
                    ]
                )
                if X_test is not None
                else None
            ),
            reconstruct=reconstruct
        )

        # computing offspring attributes
        offs.size = individual.size
        offs.nodes_collection = [new_block.nodes,*individual.nodes_collection[1:]]
        offs.nodes_count = sum(offs.nodes_collection) + offs.size - 1

        offs.depth_collection = [new_block.depth, *individual.depth_collection[1:]]
        offs.depth = max(offs.depth_collection) + offs.size - 1

        offs.id = individual.id

        return offs    
    
    return structure

# ----------------------------- STORED ----------------------------- #




# @lru_cache(maxsize=128) 
# def choose_depth_norm(max_depth, random_index, mean=None, std_dev=None):
#     """
#     Choose a depth for the structure mutation.
    
#     Parameters
#     ----------
#     max_depth : int
#         Maximum depth for generated trees.
#     random_index : list
#         List of random indices.
#     mean : float, optional
#         Mean of the normal distribution (default: None).
#     std_dev : float, optional
#         Standard deviation of the normal distribution (default: None).
        
#     Returns
#     -------
#     int
#         The chosen depth.
#     """
#     depth = max_depth - len(random_index)
#     depths = np.arange(1, depth + 1) if len(random_index) > 1 else np.arange(2, depth + 1)

#     # Ensure that depths has more than one element
#     if len(depths) == 1:
#         return depths[0]    
    
#     # Set mean and standard deviation
#     if mean is None:
#         mean = depths.mean()  # Default mean: middle of the range
#     if std_dev is None:
#         std_dev = (depths[-1] - depths[0]) / 4 
#         if std_dev == 0:
#             print("Warning: std_dev is zero")
    
#     # Generate probabilities using the normal distribution formula
#     probabilities = np.exp(-((depths - mean) ** 2) / (2 * std_dev ** 2))
#     probabilities /= probabilities.sum()  # Normalize
    
#     # Choose a depth using the probabilities
#     chosen_depth = random.choices(depths, weights=probabilities, k=1)[0]
    
#     return chosen_depth

# def structure_mutation(FUNCTIONS, TERMINALS, CONSTANTS, depth_dist="norm"):
#     """
#     Generate a function for the structure mutation.

#     Parameters
#     ----------
#     FUNCTIONS : dict
#         The dictionary of functions used in the mutation.
#     TERMINALS : dict
#         The dictionary of terminals used in the mutation.
#     CONSTANTS : dict
#         The dictionary of constants used in the mutation.
#     depth_dist : str, optional
#         Distribution to choose the depth of the new tree (default: "norm"), options: "norm", "exp", "uniform", "max", "diz"
#         If diz is chosen, then we can only decrease/increase the depth by 1 or not change it at all.
#     Returns
#     -------
    
#     Callable
#         A structure mutation function (`structure`).
        
#     """
#     def structure(individual,
#                         X,
#                         max_depth=8,
#                         p_c=0.1,
#                         p_t=0.5,
#                         X_test=None,
#                         grow_probability=1,
#                         reconstruct=True, 
#                         decay_rate=0.2,
#                         **args,
#     ):
#         """
#         Perform a mutation on a given Individual by changing the main structure of the tree.

#         Parameters
#         ----------
#         individual : Individual
#             The Individual to be mutated.
#         X : np.Tensor
#             Input data for calculating semantics.
#         max_depth : int, optional
#             Maximum depth for generated trees (default: 8).
#         p_c : float, optional
#             Probability of choosing constants (default: 0.1).
#         p_t : float, optional
#             Probability of terminal selection (default: 0.5).
#         p_prune : float, optional
#             Probability of pruning the tree (default: 0.5).
#         X_test : np.Tensor, optional
#             Test data for calculating test semantics (default: None).
#         grow_probability : float, optional
#             Probability of growing trees during mutation (default: 1). 
#             If changed, trees will be completely replaced during mutation more often.
#         replace_probability : float, optional
#             Probability of replacing the main tree during mutation (default: 0.1).
#         X_test : np.Tensor, optional
#             Test data for calculating test semantics (default: None).
#         exp_decay : bool, optional
#             Flag to indicate whether exponential decay should be used to soften the mutation (default: False).
#         reconstruct : bool
#             Whether to store the Individuals structure after mutation.

#         Returns
#         -------
#         Individual
#             The mutated individual
#         """

#         indices_with_levels = get_indices_with_levels(individual.structure[0])

#         if depth_dist == "diz": 
#             # Can only choose either an index with max depth, or the one before
#             individual_depth = individual.depth_collection[0]
#             if random.random() < 0.5:
#                 chosen_level = individual_depth - 2
#                 if chosen_level == 0:
#                     depth = random.choice([2, 3])  # Cannot choose node (1) at root
#                 elif max_depth - individual_depth > 0:
#                     depth = random.choice([1, 2, 3])
#                 else: 
#                     depth = random.choice([1, 2])
#             else:
#                 chosen_level = individual_depth - 1
#                 if max_depth - individual_depth > 0:
#                     depth = random.choice([1, 2])
#                 else:
#                     depth = random.choice([1])

#             valid_indices = [index for index, level in indices_with_levels if level == chosen_level]
#             random_index = random.choice(valid_indices)

#         else:
#             valid_indices_with_levels = [(index, level) for index, level in indices_with_levels if max_depth - level >= 2]

#             if not valid_indices_with_levels:
#                 raise ValueError("No valid indices satisfy the condition max_depth - level >= 2")

#             valid_indices, valid_levels = zip(*valid_indices_with_levels)

#             probs = exp_decay_prob(max(valid_levels) + 1, decay_rate=decay_rate)
#             level_probs = [probs[level] for level in valid_levels]
#             random_index = random.choices(valid_indices, weights=level_probs)[0]

#             if depth_dist == "norm":
#                 depth = choose_depth_norm(max_depth, random_index, mean=None, std_dev=None)
                
#             else:
#                 depth = max_depth - len(random_index)   
#                 depths = np.arange(1, depth + 1) if len(random_index) > 1 else np.arange(2, depth + 1)
                
#                 if depth_dist == "exp":
#                     probs = exp_decay_prob(len(depths), decay_rate=decay_rate)
#                     depth = random.choices(depths, weights=probs)[0]    
                    
#                 elif depth_dist == "uniform":
#                     depth = random.choice(depths)
                    
#                 elif depth_dist == "max":
#                     depth = depths[-1]
                            
        
#         # If just a node is selected
#         if depth == 1:
#             if random.random() < p_c:
#                 new_block = random.choice(list(CONSTANTS.keys()))
#             else:
#                 new_block = random.choice(list(TERMINALS.keys()))
            
#             # Swap the subtree in the main tree
#             new_structure = swap_sub_tree(individual.structure[0], new_block, list(random_index))
                        
#         else:
#             rt = get_random_tree(
#             depth,
#             FUNCTIONS,
#             TERMINALS,
#             CONSTANTS,
#             inputs=X,
#             p_c=p_c,
#             p_t=p_t,
#             grow_probability=grow_probability,
#             logistic=False,
#         )         

#             # Swap the subtree in the main tree
#             new_structure = swap_sub_tree(individual.structure[0], rt.structure, list(random_index))
    
#         # Create the new block
#         new_block = Tree(structure=new_structure,
#                             train_semantics=None,
#                             test_semantics=None,
#                             reconstruct=True)
        
#         new_block.calculate_semantics(X)            
        
#         # Create the offspring individual
#         if X_test is not None:
#             new_block.calculate_semantics(X_test, testing=True, logistic=False)
            
#         offs = Individual(
#             collection=[new_block, *individual.collection[1:]],
#             train_semantics=np.stack(
#                 [
#                     new_block.train_semantics,
#                     *individual.train_semantics[1:],
#                 ]   
#             ),
#             test_semantics=(
#                 np.stack(
#                     [
#                         new_block.test_semantics,
#                         *individual.test_semantics[1:],
#                     ]
#                 )
#                 if X_test is not None
#                 else None
#             ),
#             reconstruct=reconstruct
#         )

#         # computing offspring attributes
#         offs.size = individual.size
#         offs.nodes_collection = [new_block.nodes,*individual.nodes_collection[1:]]
#         offs.nodes_count = sum(offs.nodes_collection) + offs.size - 1

#         offs.depth_collection = [new_block.depth, *individual.depth_collection[1:]]
#         offs.depth = max(offs.depth_collection) + offs.size - 1

#         offs.id = individual.id

#         return offs    
    
#     return structure