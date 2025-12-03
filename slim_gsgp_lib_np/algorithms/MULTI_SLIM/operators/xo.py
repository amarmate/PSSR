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
Crossover operator implementation.
"""
import random

from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree import Tree
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import (get_subtree,
                                                                               replace_subtree, 
                                                                               uniform_level_choice)
from slim_gsgp_lib_np.utils.utils import get_indices_with_levels

def homologus_xo(max_depth): 
    def xo(ind1, ind2):    
        p1, p2 = ind1.collection, ind2.collection

        # Both are terminals; no crossover is possible.
        if isinstance(p1, str) and isinstance(p2, str):
            return ind1, ind2
        
        # One of them is a terminal; the terminal of one is replaced by the tree of the other. No depth restriction.
        elif isinstance(p1, tuple) and isinstance(p2, str):
            idx_lev1 = get_indices_with_levels(p1)[1:]
            idx, _ = uniform_level_choice(idx_lev1)
            offs1 = get_subtree(p1, idx)
            offs2 = replace_subtree(p1, idx, p2)
            
        elif isinstance(p1, str) and isinstance(p2, tuple):
            idx_lev2 = get_indices_with_levels(p2)[1:]
            idx, _ = uniform_level_choice(idx_lev2)
            offs1 = get_subtree(p2, idx)
            offs2 = replace_subtree(p2, idx, p1)

        # Both are tuples 
        else: 
            idx_lev1, idx_lev2 = get_indices_with_levels(p1)[1:], get_indices_with_levels(p2)[1:]
            idx_1, depth = uniform_level_choice(idx_lev1)
            same_depth = [indices for indices, d in idx_lev2 if d == depth]
            if same_depth: 
                idx_2 = random.choice(same_depth)
            else: 
                # The second parent has depth smaller than the first parent, we have to be careful 
                max_subtree_depth = max_depth - depth + 1 
                idx_2 = random.choice([indices for indices, d in idx_lev2 if d >= ind2.depth - max_subtree_depth + 1])
            
            tree_1 = get_subtree(p1, idx_1)
            tree_2 = get_subtree(p2, idx_2)
            offs1 = replace_subtree(p1, idx_1, tree_2)
            offs2 = replace_subtree(p2, idx_2, tree_1)

        return Tree(offs1), Tree(offs2)
    
    return xo


# ------------------------------------- Isnt enforcing max depth yet ---------------------------------------------------------------

# def crossover_operator(max_depth):
#     """
#     Returns a function that performs subtree crossover between two individuals, enforcing a maximum depth.
    
#     The operator uses the following rules:
#     - If both individuals are trees (tuples), randomly select a branch
#         in each and swap them.
#     - If one individual is a terminal (non-tuple) and the other is a tree,
#         randomly select a branch from the tree, graft the terminal into that spot,
#         and return the removed branch as the other offspring.
#     - If both individuals are terminals, no crossover is performed.

#     Parameters
#     ----------
#     max_depth : int
#         Maximum depth of the trees.
        
#     Returns
#     -------
#     Callable
#     """

#     def crossover(ind1, ind2):
#         """
#         Perform subtree crossover between two individuals.
        
#         The operator uses the following rules:
#         - If both individuals are trees (tuples), randomly select a branch
#             in each and swap them.
#         - If one individual is a terminal (non-tuple) and the other is a tree,
#             randomly select a branch from the tree, graft the terminal into that spot,
#             and return the removed branch as the other offspring.
#         - If both individuals are terminals, no crossover is performed.
        
#         Parameters
#         ----------
#         ind1 : any
#             The first individual (tree or terminal).
#         ind2 : any
#             The second individual (tree or terminal).
        
#         Returns
#         -------
#         tuple
#             A pair of offspring resulting from the crossover.
        
#         Example
#         -------
#         >>> ind_1 = 'S_2'
#         >>> ind_2 = (cond1, (cond2, 'S_1', 'S_4'), (cond3, 'S_4', 'S_1'))
#         >>> offs1, offs2 = crossover(ind_1, ind_2)
#         # offs1 becomes (cond2, 'S_1', 'S_4')
#         # offs2 becomes (cond1, 'S_2', (cond3, 'S_4', 'S_1'))
#         """
#         # Both individuals are trees.
#         if isinstance(ind1, tuple) and isinstance(ind2, tuple):
#             branches1 = get_all_branches(ind1)
#             branches2 = get_all_branches(ind2)
#             if branches1 and branches2:
#                 path1 = random.choice(branches1)
#                 path2 = random.choice(branches2)
#                 subtree1 = get_subtree(ind1, path1)
#                 subtree2 = get_subtree(ind2, path2)
#                 offspring1 = replace_subtree(ind1, path1, subtree2)
#                 offspring2 = replace_subtree(ind2, path2, subtree1)
#             else:
#                 offspring1, offspring2 = ind1, ind2

#         # First individual is a tree and second is terminal.
#         elif isinstance(ind1, tuple) and not isinstance(ind2, tuple):
#             branches1 = get_all_branches(ind1)
#             if branches1:
#                 path = random.choice(branches1)
#                 subtree = get_subtree(ind1, path)
#                 offspring1 = subtree
#                 offspring2 = replace_subtree(ind1, path, ind2)
#             else:
#                 offspring1, offspring2 = ind1, ind2

#         # First individual is terminal and second is a tree.
#         elif not isinstance(ind1, tuple) and isinstance(ind2, tuple):
#             branches2 = get_all_branches(ind2)
#             if branches2:
#                 path = random.choice(branches2)
#                 subtree = get_subtree(ind2, path)
#                 offspring1 = subtree
#                 offspring2 = replace_subtree(ind2, path, ind1)
#             else:
#                 offspring1, offspring2 = ind1, ind2

#         # Both are terminals; no crossover is possible.
#         else:
#             offspring1, offspring2 = ind1, ind2

#         return Tree(offspring1), Tree(offspring2)
    
#     return crossover





