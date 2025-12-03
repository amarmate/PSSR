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
Utility functions and tree operations for genetic programming.
"""

import random
import numpy as np
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.condition import Condition


def bound_value(vector, min_val, max_val):
    """
    Constrains the values within a specific range.

    Parameters
    ----------
    vector : torch.Tensor
        Input tensor to be bounded.
    min_val : float
        Minimum value for bounding.
    max_val : float
        Maximum value for bounding.

    Returns
    -------
    torch.Tensor
        A Tensor with values bounded between min_val and max_val.
    """
    return np.clip(vector, min_val, max_val)


def flatten(data):
    """
    Flattens a nested tuple structure.

    Parameters
    ----------
    data : tuple
        Input nested tuple data structure.

    Yields
    ------
    object
        Flattened data element by element. If data is not a tuple, returns the original data itself.
    """
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def create_grow_random_tree(depth, 
                            FUNCTIONS, 
                            TERMINALS, 
                            CONSTANTS, 
                            p_c=0.3,
                            p_t=0.5, 
                            first_call=True):
    """
    Generates a random tree representation using the Grow method with a maximum specified depth.

    Parameters
    ----------
    depth : int
        Maximum depth of the tree to be created.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    TERMINALS_KEYS : list
        Precomputed list of terminal keys.
    CONSTANTS_KEYS : list
        Precomputed list of constant keys.
    FUNCTIONS_KEYS : list
        Precomputed list of function keys.
    p_c : float, optional
        Probability of choosing a constant node. Default is 0.3.
    p_t : float, optional
        Probability of choosing a terminal node. Default is 0.5.
    first_call : bool, optional
        Variable that controls whether the function is being called for the first time. Default is True.

    Returns
    -------
    tuple or str
        The generated tree representation according to the specified parameters.
    """
    
    # If depth is 1 or a terminal is selected and it's not the first call
    if (depth <= 1 or random.random() < p_t) and not first_call:
        if random.random() > p_c:
            return random.choice(list(TERMINALS.keys()))
        else:
            return random.choice(list(CONSTANTS.keys()))
    
    # If a function is selected
    else:
        node = random.choice(list(FUNCTIONS.keys()))
        if FUNCTIONS[node]["arity"] == 2:
            left_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS,
                                                   p_c, p_t, False)
            right_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS,
                                                    p_c, p_t, False)
            return (node, left_subtree, right_subtree)
        else:
            left_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS,
                                                   p_c, p_t, False)
            return (node, left_subtree)


def create_random_tree(depth_condition, max_depth, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                       p_specialist=0.5, p_t=0.5, p_c=0.3):
    """
    Generates a random ensemble tree representing an individual in the GP ensemble.
    
    The tree is structured as a conditional operator:
        (condition_tree, true_branch, false_branch)
    
    - condition_tree: generated using your existing create_grow_random_tree function.
    - true_branch / false_branch: either a further nested conditional tree or, with some probability,
      a specialist selected from SPECIALISTS.
    
    Parameters
    ----------
    depth_condition : int
        Maximum depth for the condition trees.
    max_depth : int
        Maximum depth for the ensemble trees.
    FUNCTIONS : dict
        Dictionary of function nodes (for condition trees).
    TERMINALS : dict
        Dictionary of terminal nodes (for condition trees).
    CONSTANTS : dict
        Dictionary of constant nodes (for condition trees).
    SPECIALISTS : dict
        Dictionary (or list) of specialist solutions.
    p_specialist : float, optional
        Probability of terminating a branch with a specialist rather than another conditional node.
    p_t : float, optional
        Terminal probability passed to create_grow_random_tree.
    p_c : float, optional
        Constant probability passed to create_grow_random_tree.
    
    Returns
    -------
    tuple or str
        A conditional tree or a specialist (when the branch is terminated).
    """
    # Base case: if depth is 1 or by chance we decide to return a specialist
    if max_depth <= 1 or random.random() < p_specialist:
        return random.choice(list(SPECIALISTS.keys()))
    
    # Generate a condition tree.
    # Here we choose a random depth for the condition tree (at most the current depth)
    condition_tree = Condition(create_grow_random_tree(depth_condition, FUNCTIONS, TERMINALS, CONSTANTS,
                                             p_c=p_c, p_t=p_t, first_call=True))
    
    # Recursively build the true and false branches.
    true_branch = create_random_tree(depth_condition, max_depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                                     p_specialist=p_specialist, p_t=p_t, p_c=p_c)
    false_branch = create_random_tree(depth_condition, max_depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                                      p_specialist=p_specialist, p_t=p_t, p_c=p_c)
    
    return (condition_tree, true_branch, false_branch)


def initializer(pop_size, 
                depth_condition, 
                max_depth, 
                FUNCTIONS, 
                TERMINALS, 
                CONSTANTS,
                SPECIALISTS,
                p_c=0.3, 
                p_t=0.5, 
                p_s=0.5,
                **kwargs):
    """
    Generates a list of individuals with random ensemble trees for a GP population.
    
    The individuals are binned along two axes:
      - Condition tree depth: from 0 to depth_condition (inclusive).
      - Ensemble tree depth: from 1 to max_depth.
    
    For each (condition, ensemble) bin, four modes are created by adjusting the probabilities:
      1. grow–grow: condition: p_t, p_c as given; ensemble: p_specialist as given.
      2. full (condition)–grow: condition: p_t = 0, p_c = 0; ensemble: p_specialist as given.
      3. grow–full: condition: p_t, p_c as given; ensemble: p_specialist = 0.
      4. full–full: condition: p_t = 0, p_c = 0; ensemble: p_specialist = 0.
    
    Parameters
    ----------
    pop_size : int
        Total number of individuals to generate.
    depth_condition : int
        Maximum depth for condition trees. (A minimum of 2 is enforced.)
        For example, if 5 is passed then condition depths 0,1,...,5 will be used.
    max_depth : int
        Maximum ensemble tree depth. If max_depth <= 0, only specialists will be chosen.
    FUNCTIONS : dict
        Dictionary of allowed function nodes.
    TERMINALS : dict
        Dictionary of allowed terminal symbols.
    CONSTANTS : dict
        Dictionary of allowed constant values.
    SPECIALISTS : dict
        Dictionary of specialist individuals.
    p_c : float, optional
        Constant probability for tree creation (default: 0.3).
    p_t : float, optional
        Terminal probability for tree creation (default: 0.5).
    p_s : float, optional
        Specialist termination probability for ensemble tree creation (default: 0.5).
    
    Returns
    -------
    list
        A list of tree representations (each either a conditional tree or a specialist)
        forming the initial population.
    """
    # Enforce a minimum condition depth of 2.
    if depth_condition < 2:
        depth_condition = 2

    # If max_depth <= 0, then no ensemble tree can be built: return specialists.
    if max_depth <= 1:
        return [random.choice(list(SPECIALISTS.keys())) for _ in range(pop_size)]
    
    population = []
    num_condition_bins = depth_condition - 1  # condition depths: 2 ... depth_condition
    num_ensemble_bins = max_depth            # ensemble depths: 1 ... max_depth

    # Define four modes by fixing the probabilities:
    # Each mode is a tuple: (condition_p_t, condition_p_c, ensemble_p_specialist)
    modes = [
         (p_t,      p_c,      p_s),  # grow–grow
         (0,        p_c,        p_s),  # full (condition)–grow
         (p_t,      p_c,      0),             # grow–full
         (0,        p_c,        0)              # full–full
    ]
    total_bins = num_condition_bins * num_ensemble_bins * len(modes)
    inds_per_bin = pop_size // total_bins

    # Loop over all bins.
    for cond_depth in range(2, depth_condition + 1):
        for ens_depth in range(1, max_depth + 1):
            for mode in modes:
                cond_p_t, cond_p_c, ens_p_specialist = mode
                for _ in range(inds_per_bin):
                    tree = create_random_tree(
                        cond_depth, ens_depth, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                        p_specialist=ens_p_specialist, p_t=cond_p_t, p_c=cond_p_c
                    )
                    population.append(tree)

    # If there are still fewer individuals than desired, fill with trees at maximum depths using default probabilities.
    while len(population) < pop_size:
        tree = create_random_tree(
            depth_condition, max_depth, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
            p_specialist=p_s, p_t=p_t, p_c=p_c
        )
        population.append(tree)
    
    return population[:pop_size]


def tree_depth_and_nodes(tree, SPECIALISTS, depth=1):
    """
    Recursively computes the overall depth, number of nodes, and total nodes for a tree's collection.
    
    Definitions:
      - depth: the maximum recursion level among the elements in the collection.
      - nodes: total number of elements (i.e. leaves in the flattened collection).
      - total_nodes: sum of each element’s nodes_count (for Condition objects and for specialists, 
                     using SPECIALISTS[element].nodes_count).
    
    Parameters
    ----------
    tree : tuple or any
        The tree's collection. In our new design:
          - Ensemble (conditional) nodes are tuples of the form (Condition, branch_if_true, branch_if_false).
          - GP function nodes are tuples whose first element is a key in FUNCTIONS.
          - Terminals are specialist keys (strings).
    SPECIALISTS : dict
        Dictionary of specialist objects (each having attributes `nodes_count` and `depth`).
    depth : int, optional
        The current recursion level (default: 1).
    
    Returns
    -------
    depth : int
        The maximum recursion level (depth) of the collection.
    nodes : int
        The total number of elements in the flattened collection.
    total_nodes : int
        The sum of the nodes_count values for all elements.
    """
    if isinstance(tree, tuple):
        # Recursively compute stats for all children.
        # child_stats = [tree_depth_and_nodes(child, SPECIALISTS, depth + 1) for child in tree[1:]]
        child_stats = [tree_depth_and_nodes(child, SPECIALISTS, depth + 1) for child in tree]
        depth = max(depth, max(d for d, _, _ in child_stats))
        # Nodes is the sum of all children counts.
        nodes = sum(n for _, n, _ in child_stats)
        # Total nodes is the sum of the total_nodes of all children.
        total_nodes = sum(tn for _, _, tn in child_stats)
        return depth, nodes, total_nodes
    
    elif isinstance(tree, str):
        spec = SPECIALISTS[tree]
        return depth, 1, spec.nodes_count 
    
    # It is a condition 
    else: 
        return depth, 1, tree.nodes_count

               

def _execute_tree(collection, X, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS, testing=False, predict=False):
    """
    Evaluates a tree representation using the new design.
    
    In this design:
      - Ensemble (conditional) nodes are represented as:
            (Condition_object, branch_if_true, branch_if_false)
            or Specialist_key (str)
        The Condition_object already stores its semantics (train/test) and supports predict().
      - Terminals are always specialists.
    
    Parameters
    ----------
    condition : tuple
        The tre.
    X : np.ndarray
        Input data.
    FUNCTIONS : dict
        Dictionary of allowed functions.
    TERMINALS : dict
        Dictionary mapping terminal symbols to column indices.
    CONSTANTS : dict
        Dictionary mapping constant-producing keys to functions.
    SPECIALISTS : dict
        Dictionary mapping specialist keys to specialist individuals.
    testing : bool, optional
        If True, use test semantics for conditions and specialists.
    predict : bool, optional
        If True, use the predict method for conditions and specialists.
    
    Returns
    -------
    np.ndarray
        Evaluated semantics for each sample.
    """
    # If collection is a tuple the first element is a Condition object
    if isinstance(collection, tuple):
        condition_obj = collection[0]
        if predict:
            cond_semantics = condition_obj.predict(X)
        else:
            condition_obj.calculate_semantics(X, testing)
            cond_semantics = condition_obj.test_semantics if testing else condition_obj.train_semantics
        mask = cond_semantics > 0
        true_branch = _execute_tree(collection[1], X, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS, testing, predict)
        false_branch = _execute_tree(collection[2], X, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS, testing, predict)
        return np.where(mask, true_branch, false_branch)
    
    else:
        specialist = SPECIALISTS[collection]
        if predict:
            return specialist.predict(X)
        elif testing:
            return specialist.test_semantics
        else:
            return specialist.train_semantics        

def replace_subtree(tree, path, new_subtree):
    """
    Replace the subtree at the specified path with new_subtree.

    Parameters
    ----------
    tree : tuple or any
        The original tree.
    path : list of int
        The path (list of indices) to the subtree to replace.
    new_subtree : any
        The subtree (e.g. a specialist terminal) to insert.

    Returns
    -------
    The new tree with the subtree at 'path' replaced.
    """
    if not path:
        return new_subtree
    if isinstance(tree, tuple):
        index = path[0]
        tree_list = list(tree)
        tree_list[index] = replace_subtree(tree[index], path[1:], new_subtree)
        return tuple(tree_list)
    else:
        raise ValueError("Path leads into a terminal; cannot replace further.")
    
    
def get_subtree(tree, path):
    """
    Retrieve the subtree at the given path from the tree.
    """
    if not path:
        return tree
    if isinstance(tree, tuple):
        index = path[0]
        return get_subtree(tree[index], path[1:])
    else:
        return tree
    
    
def uniform_level_choice(idx_lev):
    level = random.choices(list(set([depth for _, depth in idx_lev])))[0]
    idx = random.choice([i for i, depth in idx_lev if depth == level])
    return idx, level

    
def collect_valid_subtrees(tree):
    """
    Recursively collects all valid subtrees for hoist mutation.
    
    A valid subtree is one that is either:
      - A specialist terminal (a string), or
      - A complete ensemble node: a tuple of length 3 whose first element is a Condition object.
    
    Parameters
    ----------
    tree : tuple or any
        The tree's collection.
    
    Returns
    -------
    list
        A list of valid candidate subtrees.
    """
    candidates = []
    if isinstance(tree, tuple):
        # Check if this tuple is a complete ensemble node.
        if len(tree) == 3 and hasattr(tree[0], "repr_"):
            candidates.append(tree)
        # Recurse into every child.
        for child in tree:
            candidates.extend(collect_valid_subtrees(child))
    else:
        # If not a tuple, then it's a terminal.
        if isinstance(tree, str):
            candidates.append(tree)
    return candidates


def get_condition_indices(tree, path=None, FUNCTIONS=None):
    """
    Recursively collects the indices (paths) to condition subtrees in ensemble nodes.
    
    An ensemble (conditional) node is assumed to be a tuple of length 3 
    whose first element is the condition (and is not a GP function, i.e. not in FUNCTIONS).
    
    Parameters
    ----------
    tree : tuple or any
        The tree representation (nested tuples for internal nodes, terminals as strings).
    path : list of int, optional
        The path to the current node (used during recursion). Default is [].
    FUNCTIONS : dict, optional
        Dictionary of GP functions. A node is considered an ensemble node if its first element 
        is not in FUNCTIONS. (If None, no GP functions are assumed.)
    
    Returns
    -------
    List[List[int]]
        A list of paths, where each path is a list of indices indicating the location of a condition.
    """
    if path is None:
        path = []
    indices = []
    # Check if tree is a tuple.
    if isinstance(tree, tuple):
        # If it's an ensemble (conditional) node, we assume:
        # - It has length 3.
        # - Its first element is the condition (and should not be a key in FUNCTIONS).
        if len(tree) == 3 and (FUNCTIONS is None or not (isinstance(tree[0], str) and tree[0] in FUNCTIONS)):
            # Record the path to the condition (the condition is at index 0 of the tuple).
            indices.append(path + [0])
        # Now, traverse all children.
        # For ensemble nodes, we traverse all elements (indices 0, 1, and 2).
        # For GP function nodes (first element is in FUNCTIONS), we assume children start at index 1.
        if isinstance(tree[0], str) and FUNCTIONS is not None and tree[0] in FUNCTIONS:
            arity = FUNCTIONS[tree[0]]["arity"]
            for i in range(1, arity + 1):
                indices.extend(get_condition_indices(tree[i], path + [i], FUNCTIONS))
        else:
            # For ensemble nodes, traverse every element.
            for i in range(len(tree)):
                indices.extend(get_condition_indices(tree[i], path + [i], FUNCTIONS))
    # Terminals are not traversed.
    return indices

def get_specialist_indices(tree, path=None, depth=1, **kwargs):
    """
    Recursively collects the indices (paths) to specialist terminals in the tree.
    
    A specialist is assumed to be a terminal of type string.
    
    Parameters
    ----------
    tree : tuple or any
        The tree representation.
    path : list of int, optional
        The current path (used in recursion). Default is [].
    depth : int, 1
        The current depth of the recursion.

    Returns
    -------
    List[List[int]]
        A list of paths (each a list of integers) indicating the locations of specialist terminals.
    """
    if path is None:
        path = []
    indices = []
    if isinstance(tree, tuple):
        for i, child in enumerate(tree):
            indices.extend(get_specialist_indices(child, path + [i], depth=depth+1))
    elif isinstance(tree, str):
        indices.append((path, depth))
    return indices


def get_candidate_branch_indices(tree, path=None, FUNCTIONS=None, SPECIALISTS=None):
    """
    Recursively collect candidate branch indices (paths) where a specialist can be inserted 
    to prune the tree. Only branches that are not already specialists are candidates.

    In our representation, an ensemble (conditional) node is a tuple of length 3 
    whose first element is a condition (i.e. not a GP function in FUNCTIONS). 
    Its branches (indices 1 and 2) are potential candidate sites if they are not specialists.

    Parameters
    ----------
    tree : tuple or any
        The tree representation.
    path : list of int, optional
        The current path (default: []).
    FUNCTIONS : dict, optional
        Dictionary of GP functions.
    SPECIALISTS : dict, optional
        Dictionary of specialist individuals. A node is considered a specialist if it is a string in SPECIALISTS.

    Returns
    -------
    List[List[int]]
        A list of paths (each a list of indices) representing candidate branch positions.
    """
    if path is None:
        path = []
    candidates = []
    if isinstance(tree, tuple):
        # Check if this is an ensemble node:
        if len(tree) == 3 and (FUNCTIONS is None or not (isinstance(tree[0], str) and tree[0] in FUNCTIONS)):
            # For ensemble nodes, branch indices 1 and 2 are candidates if they are not specialists.
            if not (isinstance(tree[1], str) and SPECIALISTS is not None and tree[1] in SPECIALISTS):
                candidates.append(path + [1])
            if not (isinstance(tree[2], str) and SPECIALISTS is not None and tree[2] in SPECIALISTS):
                candidates.append(path + [2])
            # Also, traverse further into branches that are not specialists.
            if not (isinstance(tree[1], str) and SPECIALISTS is not None and tree[1] in SPECIALISTS):
                candidates.extend(get_candidate_branch_indices(tree[1], path + [1], FUNCTIONS, SPECIALISTS))
            if not (isinstance(tree[2], str) and SPECIALISTS is not None and tree[2] in SPECIALISTS):
                candidates.extend(get_candidate_branch_indices(tree[2], path + [2], FUNCTIONS, SPECIALISTS))
        # If it's a GP function node, traverse its children (children start at index 1).
        elif isinstance(tree[0], str) and FUNCTIONS is not None and tree[0] in FUNCTIONS:
            arity = FUNCTIONS[tree[0]]["arity"]
            for i in range(1, arity + 1):
                candidates.extend(get_candidate_branch_indices(tree[i], path + [i], FUNCTIONS, SPECIALISTS))
        else:
            # Otherwise, traverse all children.
            for i, child in enumerate(tree):
                candidates.extend(get_candidate_branch_indices(child, path + [i], FUNCTIONS, SPECIALISTS))
    return candidates

def get_all_branches(tree, path=None):
    """
    Recursively collect valid branch indices (paths) for a tree.
    A branch is defined as any child position (starting at index 1) in a tuple.
    
    Parameters
    ----------
    tree : any
        The tree representation.
    path : list, optional
        The current path (used for recursion).
    
    Returns
    -------
    list
        A list of paths (each a list of indices) that lead to valid branch nodes.
    """
    if path is None:
        path = []
    branches = []
    if isinstance(tree, tuple) and len(tree) > 1:
        # Collect each child index as a valid branch.
        for i in range(1, len(tree)):
            current_path = path + [i]
            branches.append(current_path)
            branches.extend(get_all_branches(tree[i], current_path))
    return branches



# --------------------------------------------- NOT IMPLEMENTED ---------------------------------------------
def random_subtree(FUNCTIONS):
    """
    Creates a function that selects a random subtree from a given tree representation.

    This function generates another function that traverses a tree representation to randomly
    select a subtree based on the arity of the functions within the tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    Callable
        A function ('random_subtree_picker') that selects a random subtree from the given tree representation.

        This function navigates the tree representation recursively, choosing a subtree based on
        probabilities determined by the overall representation of the tree.

        Parameters
        ----------
        tree : tuple
            The tree representation from which to select a subtree.
        first_call : bool, optional
            Indicates whether this is the initial call to the function. Defaults to True.
        num_of_nodes : int, optional
            The total number of nodes in the tree. Used to calculate probabilities.

        Returns
        -------
        tuple
            The randomly selected subtree (or the original node if not applicable).

    Notes
    -----
    The returned function traverses the tree representation recursively, selecting subtrees based on random
    probabilities influenced by the representation of the tree.
    """
    def random_subtree_picker(tree, first_call=True, num_of_nodes=None):
        """
        Selects a random subtree from the given tree representation.

        This function navigates the tree representation recursively, choosing a subtree based on
        probabilities determined by the overall representation of the tree.

        Parameters
        ----------
        tree : tuple
            The tree representation from which to select a subtree.
        first_call : bool, optional
            Indicates whether this is the initial call to the function. Defaults to True.
        num_of_nodes : int, optional
            The total number of nodes in the tree. Used to calculate probabilities.

        Returns
        -------
        tuple
            The randomly selected subtree (or the original node if not applicable).
        """
        if isinstance(tree, tuple):
            current_number_of_nodes = (
                num_of_nodes if first_call else len(list(flatten(tree)))
            )
            if FUNCTIONS[tree[0]]["arity"] == 2:
                if first_call:
                    subtree_exploration = (
                        1
                        if random.random()
                        < len(list(flatten(tree[1]))) / (current_number_of_nodes - 1)
                        else 2
                    )
                else:
                    p = random.random()
                    subtree_exploration = (
                        0
                        if p < 1 / current_number_of_nodes
                        else (
                            1
                            if p < len(list(flatten(tree[1]))) / current_number_of_nodes
                            else 2
                        )
                    )
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                subtree_exploration = (
                    1
                    if first_call
                    else (0 if random.random() < 1 / current_number_of_nodes else 1)
                )

            if subtree_exploration == 0:
                return tree
            elif subtree_exploration == 1:
                return (
                    random_subtree_picker(tree[1], False)
                    if isinstance(tree[1], tuple)
                    else tree[1]
                )
            elif subtree_exploration == 2:
                return (
                    random_subtree_picker(tree[2], False)
                    if isinstance(tree[2], tuple)
                    else tree[2]
                )
        else:
            return tree

    return random_subtree_picker


def substitute_subtree(FUNCTIONS):
    """
    Generates a function that substitutes a specific subtree in a tree representation with a new subtree.

    This function returns another function that can recursively traverse a tree representation to replace
    occurrences of a specified subtree with a new one, maintaining the representation and
    validity of the original tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    Callable
        A function ('substitute') that substitutes a specified subtree within the given tree representation with a new subtree.

        This function recursively searches for occurrences of the target subtree within the tree
        representation and replaces it with the new subtree when found. If the original tree
        representation is a terminal or equal to the new one, return it.

        Parameters
        ----------
        tree : tuple or str
            The tree representation in which to perform the substitution. Can be a terminal.
        target_subtree : tuple or str
            The subtree to be replaced.
        new_subtree : tuple or str
            The subtree to insert in place of the target subtree.

        Returns
        -------
        tuple
            The modified tree representation with the target subtree replaced by the new subtree.
        str
            The new tree leaf node if the original is a leaf.

    Notes
    -----
    The returned function performs replacements while preserving the tree structure based on
    the arity of the function nodes.
    """

    def substitute(tree, target_subtree, new_subtree):
        """
        Substitutes a specified subtree within the given tree representation with a new subtree.

        This function recursively searches for occurrences of the target subtree within the tree
        representation and replaces it with the new subtree when found. If the original tree
        representation is a terminal or equal to the new one, return it.

        Parameters
        ----------
        tree : tuple or str
            The tree representation in which to perform the substitution. Can be a terminal.
        target_subtree : tuple or str
            The subtree to be replaced.
        new_subtree : tuple or str
            The subtree to insert in place of the target subtree.

        Returns
        -------
        tuple
            The modified tree representation with the target subtree replaced by the new subtree.
        str
            The new tree leaf node if the original is a leaf.
        """
        if tree == target_subtree:
            return new_subtree
        elif isinstance(tree, tuple):
            if FUNCTIONS[tree[0]]["arity"] == 2:
                return (
                    tree[0],
                    substitute(tree[1], target_subtree, new_subtree),
                    substitute(tree[2], target_subtree, new_subtree),
                )
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return tree[0], substitute(tree[1], target_subtree, new_subtree)
        else:
            return tree

    return substitute


def tree_pruning(TERMINALS, CONSTANTS, FUNCTIONS, p_c=0.3):
    """
    Generates a function that reduces both sides of a tree representation to a specific depth.

    This function returns another function that can prune a given tree representation to a
    specified depth by replacing nodes with terminals or constants based on a defined probability.

    Parameters
    ----------
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float, optional
        Probability of choosing a constant node. Default is 0.3.

    Returns
    -------
    Callable
        A function ('pruning') that prunes the given tree representation to the specified depth.

        This function replaces nodes in the tree representation with terminals or constants
        if the target depth is reached, ensuring the tree representation does not exceed the
        specified depth.

        Parameters
        ----------
        tree : tuple or str
            The tree representation to be pruned.
        target_depth : int
            The depth to which the tree representation should be pruned.

        Returns
        -------
        tuple
            The pruned tree representation, which may consist of terminals, constants, or
            a modified subtree.
        str
            The pruned tree if it is a leaf.
    """
    def pruning(tree, target_depth):
        """
        Prunes the given tree representation to the specified depth.

        This function replaces nodes in the tree representation with terminals or constants
        if the target depth is reached, ensuring the tree representation does not exceed the
        specified depth.

        Parameters
        ----------
        tree : tuple or str
            The tree representation to be pruned.
        target_depth : int
            The depth to which the tree representation should be pruned.

        Returns
        -------
        tuple
            The pruned tree representation, which may consist of terminals, constants, or
            a modified subtree.
        str
            The pruned tree if it is a leaf.
        """
        if target_depth <= 1 and tree not in TERMINALS:
            return (
                np.random.choice(list(TERMINALS.keys()))
                if random.random() > p_c
                else np.random.choice(list(CONSTANTS.keys()))
            )
        elif not isinstance(tree, tuple):
            return tree
        if FUNCTIONS[tree[0]]["arity"] == 2:
            new_left_subtree = pruning(tree[1], target_depth - 1)
            new_right_subtree = pruning(tree[2], target_depth - 1)
            return tree[0], new_left_subtree, new_right_subtree
        elif FUNCTIONS[tree[0]]["arity"] == 1:
            new_left_subtree = pruning(tree[1], target_depth - 1)
            return tree[0], new_left_subtree

    return pruning
