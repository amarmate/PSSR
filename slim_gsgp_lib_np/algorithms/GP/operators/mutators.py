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
Mutator operator implementation.
"""

import random
import numpy as np

from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import (random_subtree,substitute_subtree)
from slim_gsgp_lib_np.algorithms.GP.representations.tree import Tree
                                                                                        
from slim_gsgp_lib_np.utils.utils import (
        get_subtree,
        create_grow_random_tree,
        swap_sub_tree,
    )
from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import tree_depth_and_nodes, get_indices_with_levels, random_index_at_level




# Function to perform mutation on a tree.
def mutate_tree_node(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c):
    """
    Generates a function for mutating a node within a tree representation based on a set of
    terminals, constants, and functions.

    This function returns another function that can mutate a specific node in the tree representation.
    The mutation process involves randomly choosing between modifying a terminal, constant, or function node,
    while ensuring the resulting tree representation maintains valid arity (i.e., the number of child nodes
    expected by the function node).

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree to consider during mutation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float
        Probability of choosing a constant node for mutation.

    Returns
    -------
    Callable
        A function ('m_tn') that performs subtree mutation within a tree representation.

        The mutation process involves randomly choosing between modifying a terminal, constant, or function node,
        while ensuring the resulting tree representation maintains valid arity (i.e., the number of child nodes
        expected by the function node). Depending on the maximum depth of the tree or the size of the original, the
        mutation process may only return a single node.

        Parameters
        ----------
        tree : tuple
            The tree representation to mutate.

        Returns
        -------
        tuple
            The structure of the mutated tree representation.
        str
            The node resulting from mutation

    Notes
    -----
    The returned function (`m_tn`) operates recursively to traverse the tree representation and
    randomly select a node for mutation.
    """
    def m_tn(tree):
        """
        Performs subtree mutation within a tree representation.

        The mutation process involves randomly choosing between modifying a terminal, constant, or function node,
        while ensuring the resulting tree representation maintains valid arity (i.e., the number of child nodes
        expected by the function node). Depending on the maximum depth of the tree or the size of the original, the
        mutation process may only return a single node.

        Parameters
        ----------
        tree : tuple
            The tree representation to mutate.

        Returns
        -------
        tuple
            The structure of the mutated tree representation.
        str
            The node resulting from mutation
        """
        # if the maximum depth is one or the tree is just a terminal, choose a random node
        if max_depth <= 1 or not isinstance(tree, tuple):
            if random.random() > p_c:
                return random.choice(list(TERMINALS.keys()))
                # return np.random.choice(list(TERMINALS.keys()))
            else:
                return random.choice(list(CONSTANTS.keys()))
                # return np.random.choice(list(CONSTANTS.keys()))

        if FUNCTIONS[tree[0]]["arity"] == 2:
            # node_to_mutate = np.random.randint(0, 3)
            node_to_mutate = random.choice([0, 1, 2])
        elif FUNCTIONS[tree[0]]["arity"] == 1:
            # node_to_mutate = np.random.randint(0, 2) 
            node_to_mutate = random.choice([0, 1])

        inside_m = mutate_tree_node(max_depth - 1, TERMINALS, CONSTANTS, FUNCTIONS, p_c)

        # if the first node is to be mutated
        if node_to_mutate == 0:
            new_function = random.choice(list(FUNCTIONS.keys()))
            it = 0

            # making sure the arity of the chosen function matches the arity of the function to be mutated
            while (
                FUNCTIONS[tree[0]]["arity"] != FUNCTIONS[new_function]["arity"]
                or tree[0] == new_function
            ):
                new_function = random.choice(list(FUNCTIONS.keys()))

                it += 1
                # if a new valid function was not found in 10 tries, return the original function
                if it >= 10:
                    new_function = tree[0]
                    break

            # mutating the left side of the tree
            left_subtree = inside_m(tree[1])

            # mutating the right side of the tree, if the arity is 2
            if FUNCTIONS[tree[0]]["arity"] == 2:
                right_subtree = inside_m(tree[2])
                return new_function, left_subtree, right_subtree
            # if the arity is 1, returning the new function and the modified left tree
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return new_function, left_subtree

        elif node_to_mutate == 1:
            # preserving the node in position 0 and 2 while mutating position 1
            left_subtree = inside_m(tree[1])
            if FUNCTIONS[tree[0]]["arity"] == 2:
                return tree[0], left_subtree, tree[2]
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return tree[0], left_subtree
        # if the node to mutate is in position 2
        else:
            # preserving the node in position 0 and 1 while mutating position 2
            right_subtree = inside_m(tree[2])
            return tree[0], tree[1], right_subtree

    return m_tn


def mutate_tree_subtree(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c, **kwargs):
    """
    Generates a function for performing subtree mutation within a tree representation.

    This function returns another function that can perform subtree mutation by selecting a random subtree
    in the tree representation and replacing it with a newly generated random subtree.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree to consider during mutation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float
        Probability of choosing a constant node for mutation.

    Returns
    -------
    Callable
        A function ('innee_mur') that mutates a subtree in the given tree representation by replacing a randomly
        selected subtree.

        This function selects a random subtree in the input tree representation and substitutes it
        with a newly generated random subtree of the same maximum depth. If a terminal is passed,
        returns the original.

        Parameters
        ----------
        tree1 : tuple or str
            The tree representation to mutate.
        num_of_nodes : int, optional
            The number of nodes in the tree, used for selecting a random subtree.

        Returns
        -------
        tuple
            The mutated tree representation with a new subtree
        str
            The original terminal node if the input was a terminal

    Notes
    -----
    The returned function (`inner_mut`) operates by selecting a random subtree from the input tree
    representation and replacing it with a randomly generated tree representation of the same maximum depth.
    """
    # getting the subtree substitution function and the random subtree selection function
    subtree_substitution = substitute_subtree(FUNCTIONS=FUNCTIONS)
    random_subtree_picker = random_subtree(FUNCTIONS=FUNCTIONS)

    def inner_mut(tree1, num_of_nodes=None):
        """
        Mutates a subtree in the given tree representation by replacing a randomly selected subtree.

        This function selects a random subtree in the input tree representation and substitutes it
        with a newly generated random subtree of the same maximum depth. If a terminal is passed,
        returns the original.

        Parameters
        ----------
        tree1 : tuple or str
            The tree representation to mutate.
        num_of_nodes : int, optional
            The number of nodes in the tree, used for selecting a random subtree.

        Returns
        -------
        tuple
            The mutated tree representation with a new subtree
        str
            The original terminal node if the input was a terminal
        """
        if isinstance(tree1, tuple): # if the tree is a base (gp) tree
            mutation_point = random_subtree_picker(
                tree1, num_of_nodes=num_of_nodes
            )
            # gettubg a bew subtree
            new_subtree = create_grow_random_tree(
                max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=p_c
            )
            # replacing the tree in mutation point for the new substring
            new_tree1 = subtree_substitution(
                tree1, mutation_point, new_subtree
            )
            return new_tree1
        else:
            return tree1 # if tree1 is a terminal
    return inner_mut


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mutate_tree_subtree_dc(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c, p_t):
    """
    Returns a function to perform subtree mutation with depth constraint.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the resulting tree after mutation.
    TERMINALS, CONSTANTS, FUNCTIONS : dict
        Sets of elements allowed in the trees.
    p_c : float
        Probability of selecting a constant during random tree creation.
    p_t : float
        Probability of selecting a terminal during random tree creation.

    Returns
    -------
    Callable
        A mutation function that respects tree depth limits.
    """

    def mut(tree):
        d, tree = tree.depth, tree.repr_
        level = random.choice(range(0, d))
        index = random_index_at_level(tree, level)

        max_depth_new_subtree = max_depth - level - 1 

        if max_depth_new_subtree < 1:
            if random.random() < p_c: 
                new_subtree = random.choice(list(CONSTANTS.keys()))
            else: 
                new_subtree = random.choice(list(TERMINALS.keys()))

        else: 
            new_subtree = create_grow_random_tree(
                depth=max_depth_new_subtree,
                FUNCTIONS=FUNCTIONS,
                TERMINALS=TERMINALS,
                CONSTANTS=CONSTANTS,
                p_c=p_c,
                p_t=p_t,
            )
        return swap_sub_tree(tree, new_subtree, list(index))

    return mut

def mutate_tree_point(TERMINALS, CONSTANTS, FUNCTIONS, p_c):
    """
    Returns a function to perform point mutation on a tree.

    This mutation replaces a randomly selected node (function or terminal)
    with another of the same type and arity.

    Parameters
    ----------
    TERMINALS, CONSTANTS, FUNCTIONS : dict
        Sets of elements allowed in the trees.
    p_c : float
        Probability of choosing a constant for terminal mutation.

    Returns
    -------
    Callable
        A mutation function that performs point mutation.
    """
    def mut(tree):
        d, tree = tree.depth, tree.repr_
        level = random.choice(list(range(1, d)))   
        index = random_index_at_level(tree, level)
        subtree = get_subtree(tree, list(index))

        if isinstance(subtree, tuple):
            func, *args = subtree
            arity = len(args)
            possible_funcs = [f for f, data in FUNCTIONS.items() if data['arity'] == arity and f != func]
            if not possible_funcs:
                return tree  # No replacement possible
            new_func = random.choice(possible_funcs)
            new_subtree = (new_func, *args)
        else:
            # Terminal or constant
            if random.random() < p_c and CONSTANTS:
                new_subtree = random.choice(list(CONSTANTS.keys()))
            else:
                new_subtree = random.choice(list(TERMINALS.keys()))

        return swap_sub_tree(tree, new_subtree, list(index))

    return mut


def prune_mutation_tree(TERMINALS, CONSTANTS, p_c):
    """
    Returns a function to perform prune mutation on a tree.

    This mutation selects a random subtree (i.e. a node occurrence, regardless of type)
    and replaces it with a terminal or constant. The decision to use a constant vs. a
    terminal is made according to the probability p_c.

    Parameters
    ----------
    TERMINALS, CONSTANTS : dict
        Dictionaries containing the available terminal and constant elements.
    p_c : float
        The probability of selecting a constant for the replacement.

    Returns
    -------
    Callable
        A mutation function that performs prune mutation.
    """
    def mut(tree):
        d, tree = tree.depth, tree.repr_
        level = random.choice(list(range(1, d)))
        chosen_index = random_index_at_level(tree, level)
        
        # Determine the replacement: constant (with probability p_c) or terminal.
        if random.random() < p_c:
            replacement = random.choice(list(CONSTANTS.keys()))
        else:
            replacement = random.choice(list(TERMINALS.keys()))
        
        # Replace the subtree at the selected index with the replacement.
        mutated_tree = swap_sub_tree(tree, replacement, list(chosen_index))
        return mutated_tree

    return mut


def mutate_tree_hoist(TERMINALS, CONSTANTS, p_c):
    """
    Performs branch-local hoist mutation by replacing ancestral nodes with 
    selected subtree while maintaining tree validity.
    
    Parameters
    ----------
    TERMINALS : dict
        Valid terminal symbols
    CONSTANTS : dict
        Valid constant symbols
    p_c : float
        Probability of selecting constant when replacing terminals
        
    Returns
    -------
    Callable
        A hoist mutation function operating within single branches
    """
    def get_substitution_levels(index, possible=[]):
        if len(index) == 0:
            return possible
        else: 
            possible.append(index[:-1])
            return get_substitution_levels(index[:-1], possible)

    def mutate(tree): 
        d, tree = tree.depth, tree.repr_
        level = random.choice(list(range(1, d)))
        index = random_index_at_level(tree, level)

        hoist_subtree = get_subtree(tree, list(index))
        depth_subtree, _ = tree_depth_and_nodes(hoist_subtree)

        possible_substitutions = get_substitution_levels(index, [])

        if len(possible_substitutions) == 1 and depth_subtree == 1:
            if random.random() < p_c:
                new_terminal = random.choice(list(CONSTANTS.keys()))
            else:
                new_terminal = random.choice(list(TERMINALS.keys()))
            
            return swap_sub_tree(tree, new_terminal, list(index))

        else: 
            substitution_index = random.choice(possible_substitutions) if depth_subtree > 1 else random.choice(possible_substitutions[:-1])
            return swap_sub_tree(tree, hoist_subtree, list(substitution_index))
        
    return mutate


def mutator(FUNCTIONS, TERMINALS, CONSTANTS,
                  max_depth, p_c=0.3, p_t=0.5, **kwargs):
    """
    Aggregate basic mutation operators (subtree, point, prune, hoist) 
    into a single callable that randomly applies one.

    Parameters
    ----------
    FUNCTIONS : dict
        Function nodes.
    TERMINALS : dict
        Terminal symbols.
    CONSTANTS : dict
        Constant symbols.
    max_depth : int
        Maximum depth allowed for generated trees.
    p_c : float
        Probability of using a constant during random tree generation or mutation.
    p_t : float
        Probability of using a terminal during random tree generation.

    Returns
    -------
    Callable
        A mutation function that accepts a tree and returns a mutated tree.
    """
    subtree_mut = mutate_tree_subtree_dc(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c, p_t)
    # subtree_mut_original = mutate_tree_subtree(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c)
    point_mut = mutate_tree_point(TERMINALS, CONSTANTS, FUNCTIONS, p_c)
    prune_mut = prune_mutation_tree(TERMINALS, CONSTANTS, p_c)
    hoist_mut = mutate_tree_hoist(TERMINALS, CONSTANTS, p_c)

    def mutation(tree):
        r = random.random()
        if r < 0.45:
            return Tree(subtree_mut(tree))
        elif r < 0.65:
            return Tree(point_mut(tree))
        elif r < 0.85:
            return Tree(prune_mut(tree))
        else:
            return Tree(hoist_mut(tree))

    return mutation
