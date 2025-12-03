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
from slim_gsgp_lib_np.utils.utils import swap_sub_tree, get_indices, get_subtree
from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import tree_depth


def xo_operator(
        p_struct_xo=0.5,
        mut_xo_op='rshuffle', 
        max_depth=15,
        init_depth=6,
        FUNCTIONS=None,
): 
    """
    Generate a crossover operator.
    
    Parameters
    ----------
    p_struct_xo : float
        The probability of structural crossover.
        The structural crossover will happen at the root of the individual, individual.collection[0].
    mut_xo_op : str
        The mutation crossover operator to be used.
        The options are 'rshuffle' for random shuffle crossover, ...
    max_depth : int
        The maximum depth of the trees.
    init_depth : int
        The initial depth of the trees.
    FUNCTIONS : dict
        A dictionary containing the functions to be used in the trees.
    """

    depth = tree_depth(FUNCTIONS)
    
    def struct_xo(
        parent1,
        parent2,
        X,
        X_test=None,
        reconstruct=True,
    ):
        """
        Perform structural crossover on the given Individuals.

        Parameters
        ----------
        parent1 : Individual
            The first parent Individual.
        parent2 : Individual
            The second parent Individual.
        X : np.ndarray
            Input data for calculating semantics.
        X_test : np.ndarray, optional
            Test data for calculating test semantics.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover.

        Returns
        -------
        list
            A list containing the two offspring Individuals.
        """
        
        # Crossover will happen at the root of the individual, individual.collection[0]
        p1, p2 = parent1.collection[0], parent2.collection[0]

        # Find the point to crossover
        while True:
            i1, i2 = random.choice(get_indices(p1.structure)), random.choice(get_indices(p2.structure))
            i1, i2 = list(i1), list(i2)

            # Get the subtrees
            subtree1, subtree2 = get_subtree(p1.structure, i1), get_subtree(p2.structure, i2)

            # Calculate the depth of the subtrees
            depth1, depth2 = depth(subtree1), depth(subtree2)

            # Check if the depth of the subtrees is valid
            if len(i1) + depth2 < max_depth and len(i2) + depth1 < max_depth:
                break

        # Swap the subtrees
        struct_1 = swap_sub_tree(p1.structure, subtree2, i1)
        struct_2 = swap_sub_tree(p2.structure, subtree1, i2)

        offs = []

        for struct, individual in zip([struct_1, struct_2], [parent1, parent2]):
            # Create the new tree
            new_block = Tree(
                structure=struct,
                train_semantics=None,
                test_semantics=None,
                reconstruct=True,
            )

            # Calculate semantics
            new_block.calculate_semantics(X)
            if X_test is not None:
                new_block.calculate_semantics(X_test, testing=True)

            # Create the offspring individual
            new_offs = Individual(
                collection=[new_block, *individual.collection[1:]],
                train_semantics=np.stack(
                    [new_block.train_semantics, *individual.train_semantics[1:]]
                ),
                test_semantics=(
                    np.stack(
                        [new_block.test_semantics, *individual.test_semantics[1:]]
                    )
                    if X_test is not None
                    else None
                ),
                reconstruct=reconstruct,
            )

            # Update offspring attributes
            new_offs.size = individual.size
            new_offs.nodes_collection = [new_block.nodes, *individual.nodes_collection[1:]]
            new_offs.nodes_count = sum(new_offs.nodes_collection) + new_offs.size - 1
            new_offs.depth_collection = [new_block.depth, *individual.depth_collection[1:]]
            new_offs.depth = max(new_offs.depth_collection) + new_offs.size - 1

            offs.append(new_offs)

        return offs

    
    def random_shuffle_xo(
        parent1,
        parent2,
        reconstruct=True,
        ):
        """
        Perform mutation crossover on the given Individuals, using random shuffle.
        
        Parameters
        ----------
        parent1 : Individual
            The first parent Individual.
        parent2 : Individual
            The second parent Individual.
        reconstruct : bool
            Whether to reconstruct the Individual's collection after crossover.

        Returns
        -------
        offspring1, offspring2 : tuple of Individuals
            The two offspring resulting from the crossover.
        """
        # If both parents have less than 2 elements in their collection, return them as they are.
        if len(parent1.collection) < 2 and len(parent2.collection) < 2:        
            return [parent1, parent2]
        
        # Create pool of genes
        p1, p2 = parent1.collection[1:], parent2.collection[1:]
        train_semantics1, train_semantics2 = parent1.train_semantics[1:], parent2.train_semantics[1:]
        combined_collection = p1 + p2
        combined_train_semantics = list(train_semantics1) + list(train_semantics2)

        # Shuffle the combined pools to randomize gene order
        shuffled_indices = list(range(len(combined_collection)))
        random.shuffle(shuffled_indices)

        combined_collection = [combined_collection[i] for i in shuffled_indices]
        combined_train_semantics = [combined_train_semantics[i] for i in shuffled_indices]

        # XO
        # split_point = random.randint(0, len(combined_collection))
        # This point ensures that each of the offspring has the same number of genes as the parents
        # split_point = len(parent1.collection) - 1
        split_point = len(combined_collection) // 2

        new_collection1 = [parent1.collection[0]] + combined_collection[:split_point]
        new_train_semantics1 = np.stack([parent1.train_semantics[0]] + combined_train_semantics[:split_point])

        new_collection2 = [parent2.collection[0]] + combined_collection[split_point:]
        new_train_semantics2 = np.stack([parent2.train_semantics[0]] + combined_train_semantics[split_point:])

        # Test semantics
        if parent1.test_semantics is not None and parent2.test_semantics is not None:
            test_semantics1, test_semantics2 = parent1.test_semantics[1:], parent2.test_semantics[1:]
            combined_test_semantics = list(test_semantics1) + list(test_semantics2)
            
            # Shuffle test semantics using the same indices
            combined_test_semantics = [combined_test_semantics[i] for i in shuffled_indices]
            
            # Split test semantics using the same random split point
            new_test_semantics1 = np.stack([parent1.test_semantics[0]] + combined_test_semantics[:split_point])
            new_test_semantics2 = np.stack([parent2.test_semantics[0]] + combined_test_semantics[split_point:])
        else:
            new_test_semantics1 = None
            new_test_semantics2 = None

        # Create new offspring individuals
        offspring1 = Individual(
            collection=new_collection1,
            train_semantics=new_train_semantics1,
            test_semantics=new_test_semantics1,
            reconstruct=reconstruct
        )

        offspring2 = Individual(
            collection=new_collection2,
            train_semantics=new_train_semantics2,
            test_semantics=new_test_semantics2,
            reconstruct=reconstruct
        )

        offspring1.size = len(offspring1.collection)
        offspring1.nodes_collection = [offspring1.collection[0].nodes] + [offspring1.collection[i].nodes for i in range(1, offspring1.size)]
        offspring1.nodes_count = sum(offspring1.nodes_collection)

        offspring2.size = len(offspring2.collection)
        offspring2.nodes_collection = [offspring2.collection[0].nodes] + [offspring2.collection[i].nodes for i in range(1, offspring2.size)]
        offspring2.nodes_count = sum(offspring2.nodes_collection)

        # Return the two offspring
        return offspring1, offspring2

    def xo(
            parent1,
            parent2,
            X,
            X_test=None,
            reconstruct=True,
        ):
            """
            Perform crossover on the given Individuals.

            Parameters
            ----------
            parent1 : Individual
                The first parent Individual.
            parent2 : Individual
                The second parent Individual.
            X : torch.Tensor
                Input data for calculating semantics.
            X_test : torch.Tensor
                Test data for calculating test semantics.
            reconstruct : bool
                Whether to reconstruct the Individual's collection after crossover.

            Returns
            -------
            list
                A list containing the two offspring Individuals.
            """
            
            # Structural crossover
            if random.random() < p_struct_xo:
                return struct_xo(parent1, parent2, X, X_test, reconstruct)
            
            # Mutation crossover
            else:
                return random_shuffle_xo(parent1, parent2, reconstruct) if mut_xo_op == 'rshuffle' else None
                
    return xo