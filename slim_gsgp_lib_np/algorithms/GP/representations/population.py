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
Population class implementation for evaluating genetic programming individuals.
"""
# from joblib import Parallel, delayed
# from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import _execute_tree
import numpy as np 


class Population:
    def __init__(self, pop):
        """
        Initializes a population of Trees.

        This constructor sets up the population with a list of Tree objects,
        calculating the size of the population and the total node count.

        Parameters
        ----------
        pop : List
            The list of individual Tree objects that make up the population.

        Returns
        -------
        None
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.nodes_count for ind in pop)
        self.fit, self.test_fit = None, None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each tree in the population.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Boolean indicating if the calculation is for testing semantics.

        Returns
        -------
        None
        """
        # computing the semantics for all the trees in the population
        if testing and hasattr(self, "test_semantics"):
            print("Warning: Testing semantics already calculated.")
            return
        if not testing and hasattr(self, "train_semantics"):
            print("Warning: Training semantics already calculated.")
            return
        
        [
            tree.calculate_semantics(inputs, testing)
            for tree in self.population
        ]

        # computing testing semantics, if applicable
        if testing:
            # settingulation semantics to be a list with all the semantics of all trees
            self.test_semantics = np.array([
                tree.test_semantics for tree in self.population
            ])

        else:
            # setting the population semantics to be a list with all the semantics of all trees
            self.train_semantics = np.array([
                tree.train_semantics for tree in self.population
            ])


    def calculate_errors_case(self, target):
        """
        Calculate the errors case for each individual in the population (absolute values).

        Parameters
        ----------
        y_train : torch.Tensor
            Expected output (target) values for training.

        Returns
        -------
        None
        """
        if not hasattr(self, "train_semantics"):
            raise ValueError("Training semantics not calculated. Call calculate_semantics first.")
        if hasattr(self, "errors_case"):
            print("Warning: Errors case already calculated.")
            return
                
        errors = np.abs(self.train_semantics - np.stack([target] * self.train_semantics.shape[0]))        
        self.errors_case = errors
        
    def standardize_errors(self, std_errs=False): 
        if std_errs:
            threshold = 1e-5
            mean = np.mean(self.errors_case, axis=0)
            stdev = np.std(self.errors_case, axis=0)
            
            standardized_errs = np.zeros_like(self.errors_case)
            mask = stdev > threshold
            standardized_errs[:, mask] = (self.errors_case[:, mask] - mean[mask]) / stdev[mask]
                    
            # Leave masked-out (constant) columns as zero
            self.errors_case = standardized_errs

    def calculate_mad(self): 
        """
        Calculate the Mean Absolute Deviation (MAD) for the population.

        Returns 
        -------
        None
        """
        # if not hasattr(self, "errors_case"):
        #     raise ValueError("Errors case not calculated.")
        if hasattr(self, "mad"):
            return 
        
        median_case = np.median(self.errors_case, axis=0)
        self.mad = np.median(np.abs(self.errors_case - median_case), axis=0)

    def evaluate(self, target, testing=False):
        """
        Evaluate the population using the errors per case with MSE

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        target : torch.Tensor        
            Expected output (target) values.

        Returns
        -------
        None
        """
        if testing and not hasattr(self, "test_semantics"):
            raise ValueError("Testing semantics not calculated.")
        
        elif not testing and not hasattr(self, "train_semantics"):
            raise ValueError("Training semantics not calculated.")

        # Check if errors case is already calculated
        if testing: 
            sem = self.test_semantics 
            errors = sem - np.stack([target] * sem.shape[0])
            fitness = np.sqrt(np.mean(errors**2, axis=1))
            self.test_fit = fitness
            for i, individual in enumerate(self.population):
                individual.test_fitness = fitness[i]
        
        else: 
            fitness = np.sqrt(np.mean(self.errors_case**2, axis=1))
            self.fit = fitness
            for i, individual in enumerate(self.population):
                individual.fitness = fitness[i]

    def __getitem__(self, item):
        """
        Get the individual at the specified index.

        Parameters
        ----------
        item : int
            Index of the individual to get.

        Returns
        -------
        Individual
            The individual at the specified index.
        """
        return self.population[item]
    
    def __len__(self):
        """
        Return the size of the population.

        Returns
        -------
        int
            Size of the population.
        """
        return self.size
    
    def __iter__(self):
        """
        Return an iterator over the population.

        Returns
        -------
        Iterator
            Iterator over the population.
        """
        return iter(self.population)
            

    # def evaluate(self, ffunction, X, y, n_jobs=1):
    #     """
    #     Evaluates the population given a certain fitness function, input data (X), and target data (y).

    #     Attributes a fitness tensor to the population.

    #     Parameters
    #     ----------
    #     ffunction : function
    #         Fitness function to evaluate the individuals.
    #     X : torch.Tensor
    #         The input data (which can be training or testing).
    #     y : torch.Tensor
    #         The expected output (target) values.
    #     n_jobs : int
    #         The maximum number of concurrently running jobs for joblib parallelization.

    #     Returns
    #     -------
    #     None
    #     """
    #     # Evaluates individuals' semantics
    #     y_pred = Parallel(n_jobs=n_jobs)(
    #         delayed(_execute_tree)(
    #             individual.repr_, X,
    #             individual.FUNCTIONS, individual.TERMINALS, individual.CONSTANTS
    #         ) for individual in self.population
    #     )
    #     # Evaluate fitnesses
    #     self.fit = [ffunction(y, y_pred_ind) for y_pred_ind in y_pred]

    #     # Assign individuals' fitness
    #     [self.population[i].__setattr__('fitness', f) for i, f in enumerate(self.fit)]
