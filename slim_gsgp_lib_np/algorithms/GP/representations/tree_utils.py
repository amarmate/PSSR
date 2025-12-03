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
from collections import defaultdict
import numpy as np

# def bound_value(vector, min_val, max_val):
#     """
#     Constrains the values within a specific range.

#     Parameters
#     ----------
#     vector : torch.Tensor
#         Input tensor to be bounded.
#     min_val : float
#         Minimum value for bounding.
#     max_val : float
#         Maximum value for bounding.

#     Returns
#     -------
#     torch.Tensor
#         A Tensor with values bounded between min_val and max_val.
#     """
#     return np.clip(vector, min_val, max_val)

def bound_value(vector, min_val, max_val):
    """
    Faster in-place bounding without extra array allocation.
    """
    vector.clip(-1e10, 1e10, out=vector)
    return vector

# def bound_value(output, min_val, max_val):
#     """
#     In-place clip without casting error:
#     - Read output.dtype once (O(1)).
#     - Convert float bounds into that integer type if needed.
#     """
#     dt = output.dtype
#     if np.issubdtype(dt, np.integer):
#         # Convert bounds exactly to this integer dtype
#         min_int = dt.type(np.ceil(min_val))
#         max_int = dt.type(np.floor(max_val))
#         output.clip(min_int, max_int, out=output)
#     else:
#         output.clip(min_val, max_val, out=output)
#     return output


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
                            p_cond=0, 
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
    p_cond : float, optional
        Probability of choosing a conditional node. Default is 0.
    first_call : bool, optional
        Variable that controls whether the function is being called for the first time. Default is True.

    Returns
    -------
    tuple or str
        The generated tree representation according to the specified parameters.
    """
    
    if (depth <= 1 or random.random() < p_t) and not first_call:
        if random.random() > p_c:
            return random.choice(list(TERMINALS.keys()))
        else:
            return random.choice(list(CONSTANTS.keys()))

    if p_cond > 0 and random.random() < p_cond:
        node = 'cond'
        predicate = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, p_t, 0, False)
        then_else = [create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, p_t, p_cond, False) for _ in range(2)]
        children = [predicate] + then_else

    else: 
        node = random.choice([f for f in FUNCTIONS if f != 'cond'])
        arity = FUNCTIONS[node]["arity"]
        children = [create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, p_t, 0, False)
                    for _ in range(arity)]
        
    return (node, *children)    


def create_full_random_tree(depth,
                            FUNCTIONS,
                            TERMINALS,
                            CONSTANTS,
                            p_c=0.3,
                            p_cond=0.0):
    """
    Full (complete) tree generator with controlled conditional usage.

    Parameters
    ----------
    depth : int
        Maximum depth of the tree.
    FUNCTIONS : dict
        Map name -> {"arity": int, ...}
    TERMINALS : dict
        Map name -> ...
    CONSTANTS : dict
        Map name -> ...
    p_c : float
        Leaf constant vs terminal probability.
    p_cond : float
        Probability of inserting a 'cond' node at any internal node.
        Once a non-'cond' node is picked, p_cond becomes 0 below it.

    Returns
    -------
    tuple or str
        A tree represented as nested tuples, or a terminal/constant name.
    """

    if depth <= 1:
        return (random.choice(list(TERMINALS.keys()))
                if random.random() > p_c
                else random.choice(list(CONSTANTS.keys())))

    # Decide whether to insert a conditional
    if p_cond > 0 and 'cond' in FUNCTIONS and random.random() < p_cond:
        node = 'cond'
        predicate = create_full_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, 0)
        then_else = [create_full_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, p_cond) for _ in range(2)]
        children = [predicate] + then_else

    else:
        non_cond_funcs = [f for f in FUNCTIONS if f != 'cond']
        node = random.choice(non_cond_funcs)
        arity = FUNCTIONS[node]["arity"]
        children = [create_full_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, p_cond=0)
                for _ in range(arity)]

    return (node, *children)


def create_neutral_tree(operator, CONSTANTS, **kwargs):
    """
    Generates a tree with semantics all 0 if operator is 'sum' or 1 if operator is 'product'.
    
    Parameters
    ----------
    operator : str
        The operator to be used in the tree.    
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.

    Returns
    -------
    tuple
        The generated tree representation with neutral semantics.
    """
    if operator == 'sum':
        return ('subtract', list(CONSTANTS.keys())[0], list(CONSTANTS.keys())[0])
    elif operator == 'mul':
        return ('divide', list(CONSTANTS.keys())[0], list(CONSTANTS.keys())[0])
    else:
        raise ValueError("Invalid operator. Choose either 'sum' or 'mul'.")



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


def tree_depth(FUNCTIONS):
    """
    Generates a function that calculates the depth of a given tree representation.

    This function returns another function that can be used to compute the depth
    of a tree representation, which is defined as the length of the longest path
    from the root node to a leaf node.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree representation.

    Returns
    -------
    Callable
        A function ('depth') that calculates the depth of the given tree.

        This function determines the depth by recursively computing the maximum
        depth of the left and right subtrees and adding one for the current node.

        Parameters
        ----------
        tree : tuple or str
            The tree representation for which to calculate the depth. It can also be
            a terminal node represented as a string.

        Returns
        -------
        int
            The depth of the tree.

    Notes
    -----
    The returned function traverses the tree representation recursively, determining
    the depth based on the max of the subtree depths.
    """
    def depth(tree):
        """
        Calculates the depth of the given tree.

        This function determines the depth by recursively computing the maximum
        depth of the left and right subtrees and adding one for the current node.

        Parameters
        ----------
        tree : tuple or str
            The tree representation for which to calculate the depth. It can also be
            a terminal node represented as a string.

        Returns
        -------
        int
            The depth of the tree.
        """
        if not isinstance(tree, tuple):
            return 1
        else:
            arity = FUNCTIONS[tree[0]]["arity"]
            children = tree[1:1+arity]
            child_depths = [depth(child) for child in children]
            return 1 + max(child_depths)

    return depth

def tree_depth_and_nodes(tree):
    max_depth = 0
    total_nodes = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        total_nodes += 1
        if depth > max_depth:
            max_depth = depth
        if isinstance(node, tuple):
            for child in node[1:]:
                stack.append((child, depth + 1))
    return max_depth, total_nodes



# ----------------------------- acceleration of execute tree function ------------------------------
_const_cache = {}
def get_const_array(name, length, CONSTANTS):
    key = (name, length)
    if key not in _const_cache:
        val = CONSTANTS[name](None)
        _const_cache[key] = np.full((length,), val, dtype=np.float64)
    return _const_cache[key]

def _execute_tree(repr_, X, FUNCTIONS, TERMINALS, CONSTANTS):
    """
    Evaluates a tree genotype on input vectors.

    Parameters
    ----------
    repr_ : tuple
        Tree representation.

    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.

    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.

    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.

    Returns
    -------
    np.ndarray
        Output of the evaluated tree representation for all rows in X.
    """
    if isinstance(repr_, tuple):  # Function node
        function_name = repr_[0]
        arity = FUNCTIONS[function_name]["arity"]
        children = repr_[1:1 + arity]

        # Recursively evaluate all child nodes
        child_results = [
            _execute_tree(child, X, FUNCTIONS, TERMINALS, CONSTANTS)
            for child in children
        ]

        # Apply the function to the evaluated children
        output = FUNCTIONS[function_name]["function"](*child_results)
        return bound_value(output, -1e12, 1e12)    
            
    else: 
        if repr_ in TERMINALS:
            return X[:, TERMINALS[repr_]]
        elif repr_ in CONSTANTS:
            return get_const_array(repr_, X.shape[0], CONSTANTS)
        

# ============================ IMPROVED ===========================
from collections import deque, defaultdict
def get_indices_with_levels(tree):
    indices_by_level = defaultdict(list)
    stack = deque([(tree, (), 0)]) 
    
    while stack:
        sub_tree, path, level = stack.pop()
        if not isinstance(sub_tree, tuple):
            indices_by_level[level].append(path)
        else:
            if path: 
                indices_by_level[level].append(path)
            for i, child in reversed(list(enumerate(sub_tree[1:]))):
                stack.append((child, path + (i+1,), level + 1))
    
    indices_by_level[0].append(()) 
    return dict(indices_by_level)

def random_index_at_level(tree, target_level):
    """
    Returns a random index (path) at the given level in the tree.
    """
    if target_level == 0:
        return ()

    stack = deque([(tree, (), 0)])
    candidates = []

    while stack:
        node, path, level = stack.pop()
        if level == target_level:
            candidates.append(path)
        elif isinstance(node, tuple):
            for i, child in reversed(list(enumerate(node[1:]))):
                stack.append((child, path + (i + 1,), level + 1))
    return random.choice(candidates) if candidates else ()


# def get_indices_with_levels(tree):
#     """
#     Returns a dictionary mapping each depth level to a list of index paths
#     pointing to subtrees or terminal nodes at that level.

#     Supports nodes with arbitrary arity (e.g., 1, 2, 3...).

#     Parameters
#     ----------
#     tree : tuple or terminal
#         The root node of the tree.

#     Returns
#     -------
#     dict
#         A dictionary {level: [index_paths]}.
#     """
#     indices_by_level = defaultdict(list)

#     def traverse(sub_tree, path=(), level=0):
#         if not isinstance(sub_tree, tuple):
#             indices_by_level[level].append(path)
#         else:
#             if path != ():  # don't include the root twice
#                 indices_by_level[level].append(path)
#             op, *args = sub_tree
#             for i, child in enumerate(args):
#                 traverse(child, path + (i + 1,), level + 1)

#     traverse(tree)
#     indices_by_level[0].append(())  # root
#     return dict(indices_by_level)


# ============================== IMPROVED ==============================

def get_depth(tree):
    """
    Returns the depth of the tree. Terminal nodes have depth 1.

    Parameters
    ----------
    tree : tuple or terminal
        The root node of the tree.

    Returns
    -------
    int
        Maximum depth of the tree.
    """
    if not isinstance(tree, tuple):
        return 1
    _, *args = tree
    return 1 + max(get_depth(arg) for arg in args)
