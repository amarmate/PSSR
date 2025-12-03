"""
Initialization methods for PSSR ensemble trees.

This module provides functions to generate random ensemble trees
for the initial population in PSSR evolution.
"""

from typing import Optional

import numpy as np

from pssr.core.primitives import PrimitiveSet
from pssr.pssr_model.condition import Condition
from pssr.pssr_model.ensemble_individual import EnsembleIndividual, EnsembleTreeRepr


def create_grow_random_condition(
    depth: int,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
    first_call: bool = True,
) -> tuple:
    """
    Generate a random condition tree using the Grow method.
    
    Parameters
    ----------
    depth : int
        Maximum depth for the condition tree
    primitive_set : PrimitiveSet
        Primitive set with functions, terminals, and constants
    rng : np.random.Generator
        Random number generator
    p_terminal : float
        Probability of choosing a terminal when not at max depth
    p_constant : float
        Probability of choosing a constant vs variable for terminals
    first_call : bool
        Whether this is the root call (forces function at root)
        
    Returns
    -------
    tuple or str
        Tree representation for condition
    """
    # Base case: at depth 1 or randomly choose terminal
    if (depth <= 1 or rng.random() < p_terminal) and not first_call:
        if rng.random() > p_constant:
            # Variable terminal
            terminal = primitive_set.sample_terminal(rng)
            return terminal.name
        else:
            # Constant terminal
            constant = primitive_set.sample_constant(rng)
            return constant.name
    
    # Choose a function
    func = primitive_set.sample_function(rng)
    
    if func.arity == 2:
        left = create_grow_random_condition(
            depth - 1, primitive_set, rng, p_terminal, p_constant, False
        )
        right = create_grow_random_condition(
            depth - 1, primitive_set, rng, p_terminal, p_constant, False
        )
        return (func.name, left, right)
    else:
        # Arity 1
        child = create_grow_random_condition(
            depth - 1, primitive_set, rng, p_terminal, p_constant, False
        )
        return (func.name, child)


def create_random_ensemble_tree(
    depth_condition: int,
    max_depth: int,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    p_specialist: float = 0.5,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
) -> EnsembleTreeRepr:
    """
    Generate a random ensemble tree with conditional routing to specialists.
    
    The tree structure is:
    - (Condition, true_branch, false_branch) for internal nodes
    - "S_i" for specialist terminals
    
    Parameters
    ----------
    depth_condition : int
        Maximum depth for condition trees
    max_depth : int
        Maximum depth for the ensemble tree
    primitive_set : PrimitiveSet
        Primitive set with specialists, functions, terminals, and constants
    rng : np.random.Generator
        Random number generator
    p_specialist : float
        Probability of creating a specialist terminal vs conditional node
    p_terminal : float
        Probability of choosing a terminal in condition trees
    p_constant : float
        Probability of choosing a constant in condition trees
        
    Returns
    -------
    EnsembleTreeRepr
        Random ensemble tree
    """
    # Base case: at max depth or randomly choose to create specialist
    if max_depth <= 1 or rng.random() < p_specialist:
        return primitive_set.sample_specialist_name(rng)
    
    # Create a condition tree
    condition_tree = create_grow_random_condition(
        depth_condition, primitive_set, rng, p_terminal, p_constant, True
    )
    condition = Condition(condition_tree)
    
    # Recursively create branches
    true_branch = create_random_ensemble_tree(
        depth_condition, max_depth - 1, primitive_set, rng,
        p_specialist, p_terminal, p_constant
    )
    false_branch = create_random_ensemble_tree(
        depth_condition, max_depth - 1, primitive_set, rng,
        p_specialist, p_terminal, p_constant
    )
    
    return (condition, true_branch, false_branch)


def ensemble_initializer(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    population_size: int,
    depth_condition: int = 3,
    max_depth: int = 4,
    p_specialist: float = 0.5,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
) -> list[EnsembleIndividual]:
    """
    Generate an initial population of ensemble individuals.
    
    Uses a binned approach similar to ramped half-and-half:
    - Bins by condition depth (2 to depth_condition)
    - Bins by ensemble depth (1 to max_depth)
    - Four modes per bin:
      1. grow-grow: normal probabilities
      2. full-grow: p_terminal=0 for conditions, normal for ensemble
      3. grow-full: normal for conditions, p_specialist=0 for ensemble
      4. full-full: p_terminal=0 for conditions, p_specialist=0 for ensemble
    
    Parameters
    ----------
    primitive_set : PrimitiveSet
        Primitive set with specialists
    rng : np.random.Generator
        Random number generator
    population_size : int
        Number of individuals to create
    depth_condition : int
        Maximum depth for condition trees
    max_depth : int
        Maximum depth for ensemble trees
    p_specialist : float
        Base probability of creating specialist terminals
    p_terminal : float
        Base probability of choosing terminals in conditions
    p_constant : float
        Probability of choosing constants in conditions
        
    Returns
    -------
    list[EnsembleIndividual]
        Initial population of ensemble individuals
    """
    if not primitive_set.has_specialists():
        raise ValueError("Primitive set must have specialists set.")
    
    # Ensure minimum depths
    if depth_condition < 2:
        depth_condition = 2
    
    # If max_depth <= 1, only create specialists
    if max_depth <= 1:
        population = []
        for _ in range(population_size):
            specialist_name = primitive_set.sample_specialist_name(rng)
            ind = EnsembleIndividual(specialist_name, primitive_set)
            population.append(ind)
        return population
    
    population = []
    
    # Calculate bins
    num_condition_bins = depth_condition - 1  # depths 2 to depth_condition
    num_ensemble_bins = max_depth  # depths 1 to max_depth
    
    # Four modes with different probability settings
    modes = [
        (p_terminal, p_constant, p_specialist),      # grow-grow
        (0.0, p_constant, p_specialist),             # full-grow (conditions)
        (p_terminal, p_constant, 0.0),               # grow-full (ensemble)
        (0.0, p_constant, 0.0),                      # full-full
    ]
    
    total_bins = num_condition_bins * num_ensemble_bins * len(modes)
    individuals_per_bin = max(1, population_size // total_bins)
    
    # Generate individuals for each bin
    for cond_depth in range(2, depth_condition + 1):
        for ens_depth in range(1, max_depth + 1):
            for mode in modes:
                mode_p_terminal, mode_p_constant, mode_p_specialist = mode
                
                for _ in range(individuals_per_bin):
                    tree = create_random_ensemble_tree(
                        depth_condition=cond_depth,
                        max_depth=ens_depth,
                        primitive_set=primitive_set,
                        rng=rng,
                        p_specialist=mode_p_specialist,
                        p_terminal=mode_p_terminal,
                        p_constant=mode_p_constant,
                    )
                    ind = EnsembleIndividual(tree, primitive_set)
                    population.append(ind)
    
    # Fill remaining slots with random trees at max depth
    while len(population) < population_size:
        tree = create_random_ensemble_tree(
            depth_condition=depth_condition,
            max_depth=max_depth,
            primitive_set=primitive_set,
            rng=rng,
            p_specialist=p_specialist,
            p_terminal=p_terminal,
            p_constant=p_constant,
        )
        ind = EnsembleIndividual(tree, primitive_set)
        population.append(ind)
    
    # Trim to exact size
    return population[:population_size]


def fetch_ensemble_initializer(
    depth_condition: int = 3,
    max_depth: int = 4,
    p_specialist: float = 0.5,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
):
    """
    Factory function to create an ensemble initializer with preset parameters.
    
    Parameters
    ----------
    depth_condition : int
        Maximum depth for condition trees
    max_depth : int
        Maximum depth for ensemble trees
    p_specialist : float
        Probability of creating specialist terminals
    p_terminal : float
        Probability of choosing terminals in conditions
    p_constant : float
        Probability of choosing constants in conditions
        
    Returns
    -------
    Callable
        Initializer function that takes (primitive_set, rng, population_size)
    """
    def initializer(
        primitive_set: PrimitiveSet,
        rng: np.random.Generator,
        population_size: int,
        max_depth: int = max_depth,
    ) -> list[EnsembleIndividual]:
        return ensemble_initializer(
            primitive_set=primitive_set,
            rng=rng,
            population_size=population_size,
            depth_condition=depth_condition,
            max_depth=max_depth,
            p_specialist=p_specialist,
            p_terminal=p_terminal,
            p_constant=p_constant,
        )
    
    return initializer

