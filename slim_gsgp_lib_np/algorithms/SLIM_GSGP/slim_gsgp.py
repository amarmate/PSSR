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
SLIM_GSGP Class for Evolutionary Computation using PyTorch.
"""

import random
import time
import numpy as np
from slim_gsgp_lib_np.algorithms.GP.representations.tree import Tree as GP_Tree
from slim_gsgp_lib_np.algorithms.GSGP.representations.tree import Tree
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.representations.population import Population
from slim_gsgp_lib_np.utils.diversity import gsgp_pop_div_from_vectors
from slim_gsgp_lib_np.utils.logger import logger
from functools import lru_cache

class SLIM_GSGP:
    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        inflate_mutator,
        deflate_mutator,
        structure_mutator,
        xo_operator,
        verbose_reporter,
        ms,
        find_elit_func,
        p_xo=0.2,
        p_inflate=0.3,
        p_deflate=0.6,
        p_struct=0.1,
        decay_rate=0.2,
        pop_size=100,
        seed=0,
        operator="sum",
        slim_version="SLIM+SIG2",
        two_trees=True,
        p_struct_xo=0.5, 
        mut_xo_operator='rshuffle',
        settings_dict=None,
        callbacks=None,
        timeout=None,
    ):
        """
        Initialize the SLIM_GSGP algorithm with given parameters.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        inflate_mutator : Callable
            Function for inflate mutation.
        deflate_mutator : Callable
            Function for deflate mutation.
        structure_mutator : Callable
            Function for structure mutation.
        verbose_reporter : Callable
            Function to report verbose information.
        xo_operator : Callable
            Crossover operator.
        ms : Callable
            Mutation step function.
        find_elit_func : Callable
            Function to find elite individuals.
        p_xo : float
            Probability of crossover. Default is 0.
        p_inflate : float
            Probability of inflate mutation. Default is 0.3.
        p_deflate : float
            Probability of deflate mutation. Default is 0.6.
        p_struct : float
            Probability of structure mutation. Default is 0.1.
        decay_rate : float
            Decay rate for exponential decay. Default is 0.2.
        pop_size : int
            Size of the population. Default is 100.
        seed : int
            Random seed for reproducibility. Default is 0.
        operator : {'sum', 'prod'}
            Operator to apply to the semantics, either "sum" or "prod". Default is "sum".
        slim_version : str
            Version of the SLIM algorithm. Default is "SLIM+SIG2".
        two_trees : bool
            Indicates if two trees are used. Default is True.
        p_struct_xo : float
            Probability of structure crossover. Default is 0.5.
        mut_xo_operator : str
            Mutation operator for crossover. Default is 'rshuffle
        settings_dict : dict
            Additional settings passed as a dictionary.
        callbacks : list
            List of callbacks to be executed during the evolution process. Default is None.
        timeout : int
            Timeout for the evolution process. Default is None.

        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_inflate = p_inflate
        self.p_deflate = p_deflate
        self.p_struct = p_struct
        self.inflate_mutator = inflate_mutator
        self.deflate_mutator = deflate_mutator
        self.structure_mutator = structure_mutator
        self.xo_operator = xo_operator
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.operator = operator
        self.two_trees = two_trees
        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func
        self.p_struct_xo = p_struct_xo
        self.mut_xo_operator = mut_xo_operator
        self.verbose_reporter = verbose_reporter
        self.callbacks = callbacks if callbacks is not None else []
        self.stop_training = False
        self.decay_rate = decay_rate
        self.slim_version = slim_version    
        self.timeout = timeout
        self.iteration = 0

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

        GP_Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        GP_Tree.TERMINALS = pi_init["TERMINALS"]
        GP_Tree.CONSTANTS = pi_init["CONSTANTS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        run_info,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=None,
        max_depth=17,
        n_elites=1,
        reconstruct=True,
        ):
        
        """
        Solve the optimization problem using SLIM_GSGP.

        Parameters
        ----------
        X_train : array-like
            Training input data.
        X_test : array-like
            Testing input data.
        y_train : array-like
            Training output data.
        y_test : array-like
            Testing output data.
        curr_dataset : str or int
            Identifier for the current dataset.
        run_info : dict
            Information about the current run.
        n_iter : int
            Number of iterations. Default is 20.
        elitism : bool
            Whether elitism is used during evolution. Default is True.
        log : int or str
            Logging level (e.g., 0 for no logging, 1 for basic, etc.). Default is 0.
        verbose : int
            Verbosity level for logging outputs. Default is 0.
        test_elite : bool
            Whether elite individuals should be tested. Default is False.
        log_path : str
            File path for saving log outputs. Default is None.
        ffunction : function
            Fitness function used to evaluate individuals. Default is None.
        max_depth : int
            Maximum depth for the trees. Default is 17.
        n_elites : int
            Number of elite individuals to retain during selection. Default is True.
        reconstruct : bool
            Indicates if reconstruction of the solution is needed. Default is True.

        """

        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        # setting the seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting time count
        start = time.time()

        # creating the initial population
        population = Population(
            [
                Individual(
                    collection=[
                        Tree(
                            tree,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True,
                        )
                    ],
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )
        population.set_unique_id()
        
        # calculating initial population semantics
        population.calculate_semantics(X_train)
        population.calculate_errors_case(y_train, operator=self.operator)
        population.evaluate(ffunction, y=y_train, operator=self.operator, mode='fast') 

        end = time.time()

        # setting up the elite(s)
        self.elites, self.elite = self.find_elit_func(population, n_elites)   
        self.population = population

        # setting up log paths and run info
        self.log_level = log
        self.log_path = log_path
        self.run_info = run_info
        self.dataset = curr_dataset
        self.time_dict = {'struct':[], 'inflate':[], 'deflate':[], 'xo':[]}

        self.lex_rounds_history = [0] if self.selector.__name__ == "els" else None 

        # calculating the testing semantics and the elite's testing fitness if test_elite is true
        if test_elite:
            self.elite.version = self.slim_version
            pred_elite = self.elite.predict(X_test)
            self.elite.test_fitness = ffunction(pred_elite, y_test)
            # self.elite.calculate_semantics(X_test, testing=True)
            # self.elite.evaluate(
            #     ffunction, y=y_test, testing=True, operator=self.operator
            # )


        # Display and log results
        self.print_results(0, start, end) if verbose > 0 else None
        self.log_results(0, start, end)

        # Run callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)


        start_time = time.time()
    
        # begining the evolution process
        for it in range(1, n_iter + 1, 1):
            if self.selector.__name__ == "els":
                self.lex_rounds_history = []

            self.time_dict = {'struct':[], 'inflate':[], 'deflate':[], 'xo':[]}
            self.iteration += 1
            
            if time.time() - start_time > self.timeout:
                print(f"Timeout reached at iteration {it}. Training stopped.") if verbose > 0 else None
                break
            
            # starting an empty offspring population
            offs_pop, start = [], time.time()

            # adding the elite to the offspring population, if applicable
            if elitism:
                offs_pop.extend(self.elites)

            # Run callbacks
            for callback in self.callbacks:
                callback.on_generation_start(self, it)

            # filling the offspring population
            while len(offs_pop) < self.pop_size:
                r = random.random()
                if r < self.p_xo:
                    offs = self.crossover_step(population, X_train, X_test, reconstruct)
                    offs_pop.extend(offs)
                else:
                    p1 = self.selector(population)
                    if self.selector.__name__ == "els":
                        p1, i = p1
                        self.lex_rounds_history.append(i)

                    if r < self.p_inflate + self.p_xo:
                        off1 = self.inflate_mutation_step(p1, X_train, X_test, reconstruct, max_depth)
                        
                    elif r < self.p_inflate + self.p_xo + self.p_struct:
                        off1 = self.struct_mutation_step(p1, X_train, X_test, reconstruct)
                        
                    else:
                        off1 = self.deflate_mutation_step(p1, X_train, X_test, reconstruct)
                    offs_pop.append(off1)

            # removing any excess individuals from the offspring population
            if len(offs_pop) > population.size:
                offs_pop = offs_pop[: population.size]

            # turning the offspring population into a Population
            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)
            offs_pop.calculate_errors_case(y_train, operator=self.operator)
            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator, mode='fast')
            
            # replacing the current population with the offspring population P = P'
            population = offs_pop
            self.population = population

            end = time.time()

            # setting the new elite(s)
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # calculating the testing semantics and the elite's testing fitness if test_elite is true
            if test_elite:
                self.elite.version = self.slim_version
                pred_elite = self.elite.predict(X_test)
                self.elite.test_fitness = ffunction(pred_elite, y_test)
                # self.elite.calculate_semantics(X_test, testing=True)
                # self.elite.evaluate(
                #     ffunction, y=y_test, testing=True, operator=self.operator
                # )

            # Display and log results
            self.print_results(it, start, end) if verbose > 0 else None
            self.log_results(it, start, end)

            # Run callbacks
            for callback in self.callbacks:
                callback.on_generation_end(self, it, start, end)
                
            if self.stop_training:
                print(f"{it} iterations completed. Training stopped by callback.") if verbose > 0 else None
                break
            
        # Run callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)


    # ---------------------------------------------   Modules   -----------------------------------------------------------

    def crossover_step(self, population, X_train, X_test, reconstruct):
        start = time.time()
        p1, p2 = self.selector(population), self.selector(population)
        if self.selector.__name__ == "els":
            p1, i1 = p1
            p2, i2 = p2
            self.lex_rounds_history.extend([i1, i2])   

        while p1 == p2:
            p1, p2 = self.selector(population), self.selector(population)
        offs = self.xo_operator(p1, p2, X=X_train, X_test=X_test, reconstruct=reconstruct)
        self.time_dict['xo'].append(time.time() - start)
        return offs
    

    def inflate_mutation_step(self, p1, X_train, X_test, reconstruct, max_depth):
        ms_ = self.ms()
        
        if max_depth is not None and p1.size > 1 and p1.depth > max_depth:
            # Previously we had structure mutation here, but it was removed
            start = time.time()
            result = self.deflate_mutator(p1, reconstruct=reconstruct)
            self.time_dict['deflate'].append(time.time() - start)
            return result
                        
        start = time.time()
        off1 = self.inflate_mutator(
            p1,
            ms_,
            X_train,
            max_depth=self.pi_init["init_depth"],
            p_c=self.pi_init["p_c"],
            p_t=self.pi_init["p_t"],
           #X_test=X_test,
            X_test=None,
            reconstruct=reconstruct,
        )

        if max_depth is not None and off1.depth > max_depth:
            # Previously we had structure mutation here, but it was removed
            start = time.time()
            result = self.deflate_mutator(p1, reconstruct=reconstruct)
            self.time_dict['deflate'].append(time.time() - start)
            return result
        
        self.time_dict['inflate'].append(time.time() - start)
        return off1
    

    def deflate_mutation_step(self, p1, X_train, X_test, reconstruct):
        if p1.size == 1:
            # Previsously we had structure mutation here, but it was removed
            start = time.time()
            result = self.inflate_mutator(
                p1,
                self.ms(),
                X_train,
                max_depth=self.pi_init["init_depth"],
                p_c=self.pi_init["p_c"],
                p_t=self.pi_init["p_t"],
                # X_test=X_test,
                X_test=None,
                reconstruct=reconstruct,
            )

            self.time_dict['inflate'].append(time.time() - start)   
            return result
        
        start = time.time()
        result = self.deflate_mutator(p1, reconstruct=reconstruct)
        self.time_dict['deflate'].append(time.time() - start)
        return result
    
    def struct_mutation_step(self, p1, X_train, X_test, reconstruct):
        start = time.time()
        result = self.structure_mutator(
                    individual=p1,
                    X=X_train,
                    max_depth=self.pi_init["init_depth"],
                    p_c=self.pi_init["p_c"],
                    p_t=self.pi_init["p_t"],
                    # X_test=X_test,
                    X_test=None,
                    reconstruct=reconstruct,
                    decay_rate=self.decay_rate,
                    exp_decay=False,)
        self.time_dict['struct'].append(time.time() - start)
        return result


    def log_results(self, 
                    iteration, 
                    start_time, 
                    end_time,):
        
        if self.log_level == 0:
            return
                
        if self.log_level in [2, 4]:
            gen_diversity = self.calculate_diversity(iteration)
        
        if self.log_level == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(gen_diversity),
                np.std(self.population.fit),
                self.log_level,
            ]
        elif self.log_level == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                " ".join([str(ind.nodes_count) for ind in self.population.population]),
                " ".join([str(f) for f in self.population.fit]),
                self.log_level,
            ]
        elif self.log_level == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(gen_diversity),
                np.std(self.population.fit),
                " ".join([str(ind.nodes_count) for ind in self.population.population]),
                " ".join([str(f) for f in self.population.fit]),
                self.log_level,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.nodes_count, self.log_level]

        logger(
            self.log_path,
            iteration,
            self.elite.fitness,
            end_time - start_time,
            float(self.population.nodes_count),
            additional_infos=add_info,
            run_info=self.run_info,
            seed=self.seed,
        )


    def print_results(self, iteration, start, end):
                stats_data = {
                    "dataset": self.dataset,
                    "it": iteration,
                    "train": self.elite.fitness,
                    "test": self.elite.test_fitness,
                    "time": end - start,
                    "nodes": self.elite.nodes_count,
                    "div": int(self.calculate_diversity(iteration)),
                    # "avgSize": np.mean([ind.size for ind in self.population.population]),
                    # "avgFit": np.mean(self.population.fit),
                    # "std_fit": np.std(self.population.fit),   
                    "avgStru": np.mean([ind.depth_collection[0] for ind in self.population.population]),
                    "avgDep": np.mean([ind.depth for ind in self.population.population]),    
                    "struct": f"{np.round(1000*np.mean([self.time_dict['struct']]),2) if self.time_dict['struct'] != [] else 'N/A'} ({len(self.time_dict['struct'])})",
                    "inflate": f"{np.round(1000*np.mean([self.time_dict['inflate']]),2) if self.time_dict['inflate'] != [] else 'N/A'} ({len(self.time_dict['inflate'])})",
                    "deflate": f"{np.round(1000*np.mean([self.time_dict['deflate']]),2) if self.time_dict['deflate'] != [] else 'N/A'} ({len(self.time_dict['deflate'])})",
                    "xo": f"{np.round(1000*np.mean([self.time_dict['xo']]),2) if self.time_dict['xo'] != [] else 'N/A'} ({len(self.time_dict['xo'])})",
                }

                if self.lex_rounds_history:
                    stats_data["lex_r"] = np.mean(self.lex_rounds_history)

                self.verbose_reporter(
                    stats_data,
                    col_width=14,
                    first=iteration == 0,
                )


    @lru_cache(maxsize=None)
    def calculate_diversity(self, it):
        if self.operator == "sum":
            return gsgp_pop_div_from_vectors(
                np.stack([np.sum(ind.train_semantics, axis=0) for ind in self.population.population])
            )
        else:
            return gsgp_pop_div_from_vectors(
                np.stack([np.prod(ind.train_semantics, axis=0) for ind in self.population.population])
            )