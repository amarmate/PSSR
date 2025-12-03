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
Simplifier Functions for SLIM GSGP.
"""

import numpy as np
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.mutators import deflate_mutation
from slim_gsgp_lib_np.evaluators.fitness_functions import rmse

def simplify_individual(individual,
                        y_train,
                        X_train=None,
                        ffunction=None,
                        op=None,
                        reconstruct=True,
                        threshold=0.01):
    """
    Simplify an Individual by removing blocks with a fitness below a given threshold.

    Parameters
    ----------
    individual : Individual
        The individual to be simplified.
    y_train : torch.Tensor
        The training set.
    X_train : torch.Tensor
        The training set.
    ffunction : function
        The fitness function. If None, RMSE is used.
    op : str
        The operator to be used in the evaluation.. If None, RMSE is used.
    reconstruct : bool
        If True, the individual is reconstructed after removing blocks.
    threshold : float
        The threshold below which blocks are removed.

    Returns
    -------
    Individual
        The simplified individual
    """
    assert X_train is not None or ffunction is not None, "If ffunction is None, X_train must be provided."
    assert X_train is not None or op is not None, "If op is None, X_train must be provided."
    if ffunction is None:
        version = individual.version

    def best_mut_point(parent):
        """
        Find the best mutation point to remove a block from the individual.

        Parameters
        ----------
        parent : Individual
            The individual to be simplified.

        Returns
        -------
        int
            The best mutation point to remove a block.
        """
        if ffunction is None:
            preds = parent.predict(X_train)
            current_fitness = rmse(preds, y_train)
        else:
            current_fitness = parent.fitness
        fitness_diff = []

        for mut_point in range(parent.size):
            offs = deflate_mutation(parent, reconstruct=reconstruct, mut_point_idx=mut_point)

            if ffunction is None:
                offs.version = version
                preds = offs.predict(X_train)
                fit = rmse(preds, y_train)
            else:
                offs.evaluate(ffunction, y_train, testing=False, operator=op)
                fit = offs.fitness

            fit_diff = fit - current_fitness
            per_diff = fit_diff / current_fitness
            fitness_diff.append(per_diff)

        # Sort, if the lowest difference is lower than the threshold, return the mutation point
        sorted_diff = np.argsort(fitness_diff)

        if fitness_diff[sorted_diff[0]] < threshold:
            return sorted_diff[0]
        else:
            return None
        

    while True:
        try:
            mut_point = best_mut_point(individual)
        except:
            mut_point = None

        if mut_point is None:
            break
        individual = deflate_mutation(individual, reconstruct=reconstruct, mut_point_idx=mut_point)

        if ffunction is None:
            individual.version = version
            preds = individual.predict(X_train)
            fit = rmse(preds, y_train)
            individual.fitness = fit
        else:
            individual.evaluate(ffunction, y_train, testing=False, operator=op, force=True)

    return individual



# ----------------- TO IMPLEMENT ----------------- #
def simplify_structure(
        individual, 
        y_train,
        X_train=None,
        ffunction=None,
        op=None,
        reconstruct=True,
        threshold=0.01):
    
    """
    Simplify an Individual by removing branches of its GP tree

    Parameters
    ----------
    individual : Individual
        The individual to be simplified.
    y_train : torch.Tensor
        The training set.
    X_train : torch.Tensor
        The training set.
    ffunction : function    
        The fitness function. If None, RMSE is used.
    op : str
        The operator to be used in the evaluation.. If None, RMSE is used.
    reconstruct : bool
        If True, the individual is reconstructed after removing blocks.
    threshold : float
        The threshold below which blocks are removed.
        
    Returns
    -------
    Individual
        The simplified individual
    """

    def best_branch_index(parent):
        """
        Find the best branch to remove from the individual.

        Parameters
        ----------

        parent : Individual
            The individual to be simplified.
            
        Returns
        -------
        int
            The best branch to remove.
        """

        pass

    while True:
        try:
            branch_index = best_branch_index(individual)
        except:
            branch_index = None

        if branch_index is None:
            break
        
        # MUTATE 
        pass 