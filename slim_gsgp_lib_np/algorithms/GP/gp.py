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
Genetic Programming (GP) module.
"""

import random
import time

import numpy as np
from slim_gsgp_lib_np.algorithms.GP.representations.population import Population
from slim_gsgp_lib_np.algorithms.GP.representations.tree import Tree
from slim_gsgp_lib_np.utils.diversity import niche_entropy
from slim_gsgp_lib_np.utils.logger import logger
from slim_gsgp_lib_np.utils.diversity import (gsgp_pop_div_from_vectors_var)
from slim_gsgp_lib_np.utils.utils import verbose_reporter

# from functions.misc_functions import pf_rmse_comp_extended
# from functions.metrics_test import rmse

class GP:
    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        std_errs,
        mutator,
        crossover,
        find_elit_func,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        seed=0,
        settings_dict=None,
        callbacks=None, 
        elite_tree=None,
    ):
        """
        Initialize the Genetic Programming algorithm.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        mutator : Callable
            Function to mutate individuals.
        crossover : Callable
            Function to perform crossover between individuals.
        find_elit_func : Callable
            Function to find elite individuals.
        p_m : float, optional
            Probability of mutation. Default is 0.2.
        p_xo : float, optional
            Probability of crossover. Default is 0.8.
        pop_size : int, optional
            Size of the population. Default is 100.
        seed : int, optional
            Seed for random number generation. Default is 0.
        settings_dict : dict, optional
            Additional settings dictionary.
        callbacks : list, optional
            List of callbacks to be executed during the evolutionary process
        elite_tree : List of trees, optional
            List object representing the elite individuals to add to the population.
        """
        self.pi_init = pi_init
        self.selector = selector
        self.std_errs = std_errs
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict
        self.callbacks = callbacks if callbacks is not None else []
        self.hall_of_fame = []

        if elite_tree != None:
            self.elite_tree = [Tree(tree.repr_) for tree in elite_tree]
        else:
            self.elite_tree = None

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        elitism=True,
        log_level=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        max_depth=None,
        ffunction=None,
        n_elites=1,
        depth_calculator=None,
        n_jobs=1,
        it_tolerance=500, 
    ):
        """
        Execute the Genetic Programming algorithm.
        """
        # Set seeds for reproducibility.
        np.random.seed(self.seed)
        random.seed(self.seed)

        # in-memory evaluation logging
        if log_level == "evaluate":
            self.log = {
                "generation": [],
                "time": [],
                "train_rmse": [],
                "val_rmse": [],
                "nodes_count": [],
                "diversity_var": [],
            }

        start = time.time()
        population = Population([Tree(tree) for tree in self.initializer(**self.pi_init)])
        
        if self.elite_tree:
            [population.population.pop() for _ in range(len(self.elite_tree))]
            population.population.extend(self.elite_tree)

        population.calculate_semantics(inputs=X_train, testing=False)
        population.calculate_errors_case(y_train)
        population.evaluate(target=y_train, testing=False)
        population.standardize_errors(self.std_errs)
        self.population = population
        self.dataset = curr_dataset

        end = time.time()
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # Test the elite on the testing data if required.
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        # Logging initial generation.
        if log_level:
            self.log_generation(0, population, end - start, log_level, log_path, run_info)

        # Initialize the time tracker for mutation and crossover.
        self.time_dict = {'mutation': [], 'xo': []}
        if self.selector.__name__ in ["els", "mels"]:
            self.lex_rounds = [0]

        # Verbose reporting.
        self.print_results(0, start, end) if verbose != 0 else None

        for callback in self.callbacks:
            callback.on_train_start(self)

        # ------------------------- EVOLUTIONARY PROCESS -------------------------
        count_tolerance, best_fitness = 0, self.elite.fitness
        for it in range(1, n_iter + 1):
            count_tolerance += 1
            self.lex_rounds = [] if self.selector.__name__ in ["els", "mels"] else None

            # Reset the timing dictionary for the new generation.
            self.time_dict = {'mutation': [], 'xo': []}

            for callback in self.callbacks:
                callback.on_generation_start(self, it)

            # Evolve the population for this generation.
            offs_pop, gen_start = self.evolve_population(
                population,
                ffunction,
                max_depth,
                depth_calculator,
                elitism,
                X_train,
                y_train,
            )
            population = offs_pop
            gen_end = time.time()

            # Get the new elite(s) and test on test data if required.
            self.elites, self.elite = self.find_elit_func(population, n_elites)
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            # Logging the generation and verbose reporting.
            if log_level:
                self.log_generation(it, population, gen_end - gen_start, log_level, log_path, run_info) 

            self.print_results(it, gen_start, gen_end) if verbose != 0 else None

            for callback in self.callbacks:
                callback.on_generation_end(self, it, gen_start, gen_end)
            
            if self.elite.fitness < best_fitness:
                best_fitness = self.elite.fitness
                count_tolerance = 0     

            if count_tolerance >= it_tolerance:
                break           
        
        for callback in self.callbacks:
            callback.on_train_end(self)

    def evolve_population(
        self,
        population,
        ffunction,
        max_depth,
        depth_calculator,
        elitism,
        X_train,
        y_train,
    ):
        """
        Evolve the population for one generation.
        """
        offs_pop = []
        gen_start = time.time()

        # Add elites if elitism is enabled.
        if elitism:
            offs_pop.extend(self.elites)
            self.hall_of_fame.extend(self.elites)

        # Fill the offspring population.
        while len(offs_pop) < self.pop_size:
            if random.random() < self.p_xo:
                offspring = self.crossover_step(population, max_depth, depth_calculator)
            else:
                offspring = [self.mutation_step(population, max_depth, depth_calculator)]
            offs_pop.extend(offspring)

        # Ensure the offspring population matches the required size.
        if len(offs_pop) > population.size:
            offs_pop = offs_pop[: population.size]
                                    
        # Evaluate the new population.
        offs_pop = Population(offs_pop)
        offs_pop.calculate_semantics(X_train, testing=False)
        offs_pop.calculate_errors_case(y_train)
        offs_pop.evaluate(target=y_train, testing=False)
        offs_pop.standardize_errors(self.std_errs)
        self.population = offs_pop
        return offs_pop, gen_start

    def crossover_step(self, population, max_depth, depth_calculator):
        """
        Perform the crossover operation while tracking its execution time.
        """
        start = time.time()

        # Select two distinct parents.
        p1 = self.selector(population)
        p2 = self.selector(population)

        if self.selector.__name__ in ["els", "mels"]:
            p1, i1 = p1
            p2, i2 = p2
            self.lex_rounds.extend([i1, i2])

        # Generate offspring from the selected parents.
        offs1, offs2 = self.crossover(
            p1,
            p2,
        )
        elapsed = time.time() - start
        self.time_dict['xo'].append(elapsed)
        return [offs1, offs2]

    def mutation_step(self, population, max_depth, depth_calculator):
        """
        Perform the mutation operation while tracking its execution time.
        """
        start = time.time()

        p1 = self.selector(population) 

        if self.selector.__name__ in ["els", "mels"]:
            p1, i1 = p1
            self.lex_rounds.append(i1)

        offs1 = self.mutator(p1)
        elapsed = time.time() - start
        self.time_dict['mutation'].append(elapsed)
        return offs1
    
    def log_generation(self, generation, population, elapsed_time, log_level, log_path, run_info):
        """
        Log the results for the current generation including mutation and crossover timings.
        """
        if not log_path or log_level == 0:
            return 
        
        # special in-memory evaluation log
        if log_level == "evaluate":
            train_rmse    = float(self.elite.fitness)
            val_rmse      = float(self.elite.test_fitness)
            nodes_count   = int(self.elite.nodes_count)
            diversity_var = float(np.std(population.fit))

            self.log["generation"].append(generation)
            self.log["time"].append(elapsed_time + self.log["time"][-1] if self.log["time"] else elapsed_time)
            self.log["train_rmse"].append(train_rmse)
            self.log["val_rmse"].append(val_rmse)
            self.log["nodes_count"].append(nodes_count)
            self.log["diversity_var"].append(diversity_var)
            return

        # Prepare additional logging info based on the log level.
        if log_level == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
            ]
        elif log_level == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                " ".join([str(ind.nodes_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
            ]
        elif log_level == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.nodes_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.nodes_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.nodes_count]

        logger(
            log_path,
            generation,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )

    def print_results(self, iteration, start, end):
        params = {
            "dataset": self.dataset,
            "it": iteration,
            "train": self.elite.fitness,
            "test": self.elite.test_fitness,
            "time": end - start,
            "nodes": self.elite.nodes_count,
            "avg_nodes": np.mean([ind.nodes_count for ind in self.population.population]),
            # "div": int(gsgp_pop_div_from_vectors(self.population.train_semantics)),
            "div (var)" : int(gsgp_pop_div_from_vectors_var(self.population.train_semantics)),
            "mut": f"{np.round(1000*np.mean([self.time_dict['mutation']]),2) if self.time_dict['mutation'] != [] else 'N/A'} ({len(self.time_dict['mutation'])})",
            "xo": f"{np.round(1000*np.mean([self.time_dict['xo']]),2) if self.time_dict['xo'] != [] else 'N/A'} ({len(self.time_dict['xo'])})",
        }
        if self.selector.__name__ in ["els", "mels"]:
            params["lex_r"] = np.mean(self.lex_rounds)
        
        verbose_reporter(
                params, 
                first = iteration == 0,
                precision=3, 
                col_width=14
        )




