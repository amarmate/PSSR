from matplotlib import pyplot as plt
import numpy as np
from functions.misc_functions import pf_rmse_comp_extended


class SLIM_GSGP_Callback:
    """
    Base class for callbacks.

    Methods
    -------
    on_train_start(slim_gsgp)
        Called at the beginning of the training process.
    on_train_end(slim_gsgp)
        Called at the end of the training process.
    on_generation_start(slim_gsgp, generation)
        Called at the beginning of each generation.
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    """

    def on_train_start(self, slim_gsgp):
        """
        Called at the beginning of the training process.
        """
        pass

    def on_train_end(self, slim_gsgp):
        """
        Called at the end of the training process.
        """
        pass

    def on_generation_start(self, slim_gsgp, iteration):
        """
        Called at the beginning of each generation.
        """
        pass

    def on_generation_end(self, slim_gsgp, iteration, start, end):
        """
        Called at the end of each generation.
        """
        pass


class LogDiversity(SLIM_GSGP_Callback):
    """
    Callback to log the diversity of the population.
    
    Attributes
    ----------
    diversity_structure : list
        List to store the diversity of the structure of the individuals.
    diversity_semantics : list
        List to store the diversity of the semantics of the individuals.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    plot_diversity()    
        Plot the diversity of the population.
    """

    def __init__(self):
        self.diversity_structure = []
        self.diversity_semantics = []

    def on_generation_end(self, slim_gsgp, generation, *args):
        individual_structure = [individual.structure[0] for individual in slim_gsgp.population.population]
        self.diversity_structure.append(len(set(individual_structure)))
        self.diversity_semantics.append(slim_gsgp.calculate_diversity(generation))

    def plot_diversity(self):
        fig, axs = plt.subplots(1,2, figsize=(18, 5))
        fig.suptitle('Diversity of the population')
        axs[0].plot(self.diversity_structure)   
        axs[0].set_title('Structure diversity')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Number of different structures')

        axs[1].plot(self.diversity_semantics)
        axs[1].set_title('Semantics diversity')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Diversity')
        plt.show()


class LogFitness(SLIM_GSGP_Callback):
    """
    Callback to log the fitness of the best individual in the population.

    Attributes
    ----------
    test_fitness : list
        List to store the test fitness of the best individual in the population.
    train_fitness : list
        List to store the train fitness of the best individual in the population.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    plot_fitness()
        Plot the fitness of the best individual in the population.
    """
    
    def __init__(self):
        self.test_fitness = []
        self.train_fitness = []

    def on_generation_end(self, slim_gsgp, generation, *args):
        self.test_fitness.append(slim_gsgp.elite.test_fitness)
        self.train_fitness.append(slim_gsgp.elite.fitness)

    def plot_fitness(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.test_fitness, label='Test fitness')
        plt.plot(self.train_fitness, label='Train fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()
        
class LogAge(SLIM_GSGP_Callback):
    """
    Callback to log the age of all the individuals in the population.

    Attributes
    ----------
    age : list
        List to store the age of all the individuals in the population in the last generation.

    Methods
    -------
    on_train_end(slim_gsgp)
        Called at the end of the training process.
    plot_age()
        Plot the age distribution of the population.    
    """
    def __init__(self):
        self.age = []

    def on_train_end(self, slim_gsgp, *args):
        self.age.append([individual.age for individual in slim_gsgp.population.population])

    def plot_age(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.age[-1], bins=20)
        plt.xlabel('Age')
        plt.ylabel('Number of individuals')
        plt.show()
        
        
        
def EarlyStopping(patience=10):
    """
    Callback to stop the training process when the fitness of the best individual does not improve for a number of generations.

    Attributes
    ----------
    patience : int
        Number of generations without improvement to wait before stopping the training process.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    """

    class EarlyStoppingCallback(SLIM_GSGP_Callback):
        def __init__(self):
            self.best_fitness = None
            self.counter = 0

        def on_generation_end(self, slim_gsgp, generation, *args):
            if generation == 1:
                # Reinicialize the counter and the best fitness
                self.best_fitness = slim_gsgp.elite.test_fitness.item()
                self.counter = 0
                
            elif self.best_fitness is None or slim_gsgp.elite.test_fitness.item() < self.best_fitness:
                self.best_fitness = slim_gsgp.elite.test_fitness.item()
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= patience:
                slim_gsgp.stop_training = True

    return EarlyStoppingCallback()


def EarlyStopping_train(patience=10):
    """
    Callback to stop the training process when the train fitness of the best individual in the training set does not improve for a number of generations.

    Attributes
    ----------
    patience : int
        Number of generations without improvement to wait before stopping the training process.

    Methods
    -------
    on_generation_end(slim_gsgp, generation)
        Called at the end of each generation.
    """

    class EarlyStoppingCallback(SLIM_GSGP_Callback):
        def __init__(self):
            self.best_fitness = None
            self.counter = 0

        def on_generation_end(self, slim_gsgp, generation, *args):
            if generation == 1:
                # Reinicialize the counter and the best fitness
                self.best_fitness = slim_gsgp.elite.fitness.item()
                self.counter = 0
                
            elif self.best_fitness is None or slim_gsgp.elite.fitness.item() < self.best_fitness:
                self.best_fitness = slim_gsgp.elite.fitness.item()
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= patience:
                slim_gsgp.stop_training = True

    return EarlyStoppingCallback()
    

class LogDescendance(SLIM_GSGP_Callback):
    """
    Callback to log the descendance of the individuals in the population.
    
    Attributes
    """
    def __init__(self):
        self.id_dist = []
        
    def on_generation_end(self, slim_gsgp, generation, *args):
        descendance = [individual.id for individual in slim_gsgp.population.population]
        self.id_dist.append(len(set(descendance)))
    
    def plot_descendance(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.id_dist)
        plt.xlabel('Generation')
        plt.ylabel('Number of different individuals')
        plt.show()

    
class LogSpecialist(SLIM_GSGP_Callback):
    def __init__(self, X_train, y_train, masks):
        """
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
        y_train : array-like, shape (n_samples,)
        masks   : list of boolean arrays, each of shape (n_samples,)
                  Each mask defines the data region for one specialist.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.masks = masks

        self.log_rmse, self.log_size, self.log_rmse_out, self.log_best_ensemble = [], [], [], []  

    def on_generation_start(self, optimizer, generation):
        min_errs, best_inds, sizes, errs_out = [], [], [], []
        total_sq_errs = 0

        for mask in self.masks: 
            errors_mask = optimizer.population.errors_case[:, mask]
            errors_ind = np.sqrt(np.mean(errors_mask**2, axis=1))
            best_ind = np.argmin(errors_ind)
            min_err = errors_ind[best_ind]
            errors_out = optimizer.population.errors_case[best_ind, ~mask]
            min_err_out = np.sqrt(np.mean(errors_out**2)) if np.sum(~mask) > 0 else 0

            min_errs.append(min_err)
            best_inds.append(optimizer.population.population[best_ind])
            sizes.append(optimizer.population.population[best_ind].total_nodes)
            errs_out.append(min_err_out)
            total_sq_errs += np.sum(errors_mask[best_ind] ** 2)

        self.log_rmse.append(min_errs)
        self.log_size.append(sizes)
        self.log_rmse_out.append(errs_out)
        self.log_best_ensemble.append(np.sqrt(total_sq_errs / len(self.masks[0])))

    def get_log_dict(self): 
        G = len(self.log_rmse)
        return { 
            'generation': list(range(G)),
            'specialist_rmse': np.array(self.log_rmse),
            'specialist_size': np.array(self.log_size),
            'specialist_rmse_out': np.array(self.log_rmse_out),
            'best_ensemble_rmse': np.array(self.log_best_ensemble),
        }

    def plot_specialist_fitnesses(self, best_ensemble=False):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        data_rmse = np.array(self.log_rmse) 
        for i in range(data_rmse.shape[1]):
            axs[0].plot(data_rmse[:, i], label=f"Specialist {i+1} ({data_rmse[-1, i]:.2f})")
        best_ensemble_data = np.array(self.log_best_ensemble)
        axs[0].plot(best_ensemble_data, label=f"Best Ensemble ({best_ensemble_data[-1]:.2f})", linestyle='--')
        axs[0].set_title('Specialist RMSE over Generations')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('RMSE')
        axs[0].legend()

        data_size = np.array(self.log_size)
        for i in range(data_size.shape[1]):
            axs[1].plot(data_size[:, i], label=f"Specialist {i+1} ({data_size[-1, i]:.2f})")
        axs[1].set_title('Specialist Size over Generations')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Size')
        axs[1].legend()

        plt.tight_layout()
        plt.show()  

    def plot_best_ensemble(self):
        fig, ax = plt.subplots()
        rmse = np.array(self.log_best_ensemble) 
        sizes = np.sum(self.log_size, axis=1)
        line1 = ax.plot(rmse, label=f"Best Ensemble ({rmse[-1]:.2f})")
        ax2 = ax.twinx()
        line2 = ax2.plot(sizes, label=f"Total Size ({sizes[-1]:.2f})", linestyle='-.', color='orange')
        ax2.set_ylabel('Size')
        ax.set_title('Best Ensemble RMSE and SIZE over Generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('RMSE')
        ax.legend(loc='upper left', handles=line1+line2)
        plt.show()  

    def plot_pf_ensemble(self, comparison=None): 
        rmse = np.array(self.log_best_ensemble) 
        sizes = np.sum(self.log_size, axis=1)
        points = list(zip(sizes, rmse))
        pf = np.array(pf_rmse_comp_extended(points))

        fig, ax = plt.subplots()

        if comparison is not None:
            if comparison.shape[1] != 2:
                raise ValueError("Comparison data must have two columns.")
            ax.scatter(comparison[:, 0], comparison[:, 1], marker='o', color='red')
            ax.plot(comparison[:, 0], comparison[:, 1], label='Comparison Pareto Front', color='red')
            
        ax.scatter(pf[:, 0], pf[:, 1], marker='o', color='orange')
        ax.plot(pf[:, 0], pf[:, 1], label='Pareto Front', color='orange')

        ax.set_title('Pareto Front of Ensemble')
        ax.set_xlabel('Size')
        ax.set_ylabel('RMSE')
        ax.legend()
        plt.show()

        return pf