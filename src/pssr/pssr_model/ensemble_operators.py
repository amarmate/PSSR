"""
Variation operators for PSSR ensemble trees.

This module provides mutation and crossover operators for evolving
ensemble individuals with conditional routing to specialists.
"""

from typing import Callable, Optional

import numpy as np

from pssr.core.primitives import PrimitiveSet
from pssr.core.representations.population import Population
from pssr.pssr_model.condition import Condition
from pssr.pssr_model.ensemble_individual import EnsembleIndividual, EnsembleTreeRepr
from pssr.pssr_model.ensemble_initialization import (
    create_grow_random_condition,
    create_random_ensemble_tree,
)


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_specialist(
    individual: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
) -> EnsembleIndividual:
    """
    Mutate by replacing a random specialist with another specialist.
    
    Parameters
    ----------
    individual : EnsembleIndividual
        Individual to mutate
    primitive_set : PrimitiveSet
        Primitive set with specialists
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    EnsembleIndividual
        Mutated individual
    """
    # Get all specialist positions
    specialist_indices = individual.get_specialist_indices()
    
    if not specialist_indices:
        # No specialists to mutate, return copy
        return individual.copy()
    
    # Choose random specialist position
    idx = rng.integers(0, len(specialist_indices))
    path, _ = specialist_indices[idx]
    
    # Choose new specialist (possibly the same one)
    new_specialist = primitive_set.sample_specialist_name(rng)
    
    # Replace
    return individual.replace_subtree(path, new_specialist)


def mutate_condition(
    individual: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    depth_condition: int = 3,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
) -> EnsembleIndividual:
    """
    Mutate by replacing a random condition with a new random condition.
    
    Parameters
    ----------
    individual : EnsembleIndividual
        Individual to mutate
    primitive_set : PrimitiveSet
        Primitive set
    rng : np.random.Generator
        Random number generator
    depth_condition : int
        Maximum depth for new condition
    p_terminal : float
        Terminal probability for condition generation
    p_constant : float
        Constant probability for condition generation
        
    Returns
    -------
    EnsembleIndividual
        Mutated individual
    """
    # Get all condition positions
    condition_indices = individual.get_condition_indices()
    
    if not condition_indices:
        # No conditions to mutate (tree is just a specialist)
        return individual.copy()
    
    # Choose random condition position
    idx = rng.integers(0, len(condition_indices))
    path = condition_indices[idx]
    
    # Create new condition tree
    new_condition_tree = create_grow_random_condition(
        depth_condition, primitive_set, rng, p_terminal, p_constant, True
    )
    new_condition = Condition(new_condition_tree)
    
    # Replace
    return individual.replace_subtree(path, new_condition)


def mutate_subtree(
    individual: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    depth_condition: int = 3,
    max_depth: int = 4,
    p_specialist: float = 0.5,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
) -> EnsembleIndividual:
    """
    Mutate by replacing a random branch with a new random subtree.
    
    Parameters
    ----------
    individual : EnsembleIndividual
        Individual to mutate
    primitive_set : PrimitiveSet
        Primitive set with specialists
    rng : np.random.Generator
        Random number generator
    depth_condition : int
        Maximum depth for conditions
    max_depth : int
        Maximum depth for new subtree
    p_specialist : float
        Probability of specialist terminals
    p_terminal : float
        Terminal probability for conditions
    p_constant : float
        Constant probability for conditions
        
    Returns
    -------
    EnsembleIndividual
        Mutated individual
    """
    # Get all branch positions (positions 1 and 2 in tuples)
    specialist_indices = individual.get_specialist_indices()
    
    if not specialist_indices:
        # Tree is a single specialist, replace whole tree
        new_tree = create_random_ensemble_tree(
            depth_condition, max_depth, primitive_set, rng,
            p_specialist, p_terminal, p_constant
        )
        return EnsembleIndividual(new_tree, primitive_set)
    
    # Choose a random branch position (specialist position = end of a branch)
    idx = rng.integers(0, len(specialist_indices))
    path, depth = specialist_indices[idx]
    
    # Calculate remaining depth
    remaining_depth = max(1, max_depth - depth + 1)
    
    # Create new subtree
    new_subtree = create_random_ensemble_tree(
        depth_condition, remaining_depth, primitive_set, rng,
        p_specialist, p_terminal, p_constant
    )
    
    # Replace
    return individual.replace_subtree(path, new_subtree)


def mutate_hoist(
    individual: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
) -> EnsembleIndividual:
    """
    Mutate by hoisting a random subtree to become the new root.
    
    This simplifies the tree by promoting a subtree.
    
    Parameters
    ----------
    individual : EnsembleIndividual
        Individual to mutate
    primitive_set : PrimitiveSet
        Primitive set
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    EnsembleIndividual
        Mutated individual
    """
    # Collect all valid subtrees (complete conditional nodes or specialists)
    subtrees = _collect_valid_subtrees(individual.collection)
    
    if len(subtrees) <= 1:
        # Nothing to hoist
        return individual.copy()
    
    # Choose a random subtree (excluding root)
    idx = rng.integers(1, len(subtrees))
    new_root = subtrees[idx]
    
    # Deep copy the subtree
    new_tree = _deep_copy_subtree(new_root)
    
    return EnsembleIndividual(new_tree, primitive_set)


def _collect_valid_subtrees(tree: EnsembleTreeRepr) -> list[EnsembleTreeRepr]:
    """Collect all valid subtrees (conditional nodes or specialists)."""
    candidates = [tree]
    
    if isinstance(tree, tuple):
        # Recurse into branches (not the condition)
        candidates.extend(_collect_valid_subtrees(tree[1]))
        candidates.extend(_collect_valid_subtrees(tree[2]))
    
    return candidates


def _deep_copy_subtree(tree: EnsembleTreeRepr) -> EnsembleTreeRepr:
    """Deep copy a subtree."""
    if isinstance(tree, tuple):
        condition = tree[0]
        if isinstance(condition, Condition):
            condition = condition.copy()
        return (
            condition,
            _deep_copy_subtree(tree[1]),
            _deep_copy_subtree(tree[2]),
        )
    else:
        return tree


def mutate_prune(
    individual: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
) -> EnsembleIndividual:
    """
    Mutate by pruning a random branch to a specialist.
    
    This simplifies the tree by replacing a conditional subtree with a specialist.
    
    Parameters
    ----------
    individual : EnsembleIndividual
        Individual to mutate
    primitive_set : PrimitiveSet
        Primitive set
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    EnsembleIndividual
        Mutated individual
    """
    # Get all branch positions that are not already specialists
    branches = _get_prunable_branches(individual.collection)
    
    if not branches:
        # Nothing to prune
        return individual.copy()
    
    # Choose random branch
    idx = rng.integers(0, len(branches))
    path = branches[idx]
    
    # Replace with random specialist
    new_specialist = primitive_set.sample_specialist_name(rng)
    
    return individual.replace_subtree(path, new_specialist)


def _get_prunable_branches(tree: EnsembleTreeRepr, path: Optional[list] = None) -> list[list[int]]:
    """Get paths to branches that can be pruned (non-specialist subtrees)."""
    if path is None:
        path = []
    
    branches = []
    
    if isinstance(tree, tuple):
        # Check branches 1 and 2
        for i in [1, 2]:
            branch = tree[i]
            if isinstance(branch, tuple):
                # This branch is a conditional subtree, can be pruned
                branches.append(path + [i])
                # Recurse
                branches.extend(_get_prunable_branches(branch, path + [i]))
    
    return branches


# =============================================================================
# Aggregated Mutation Operator
# =============================================================================

def ensemble_mutator(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    depth_condition: int = 3,
    max_depth: int = 4,
    p_specialist_mut: float = 0.3,
    p_condition_mut: float = 0.3,
    p_subtree_mut: float = 0.2,
    p_hoist_mut: float = 0.1,
    p_prune_mut: float = 0.1,
    p_terminal: float = 0.5,
    p_constant: float = 0.3,
    p_specialist: float = 0.5,
) -> Callable[[EnsembleIndividual], EnsembleIndividual]:
    """
    Create an aggregated mutation operator for ensemble individuals.
    
    Parameters
    ----------
    primitive_set : PrimitiveSet
        Primitive set with specialists
    rng : np.random.Generator
        Random number generator
    depth_condition : int
        Maximum depth for condition trees
    max_depth : int
        Maximum depth for ensemble trees
    p_specialist_mut : float
        Probability of specialist swap mutation
    p_condition_mut : float
        Probability of condition mutation
    p_subtree_mut : float
        Probability of subtree mutation
    p_hoist_mut : float
        Probability of hoist mutation
    p_prune_mut : float
        Probability of prune mutation
    p_terminal : float
        Terminal probability for conditions
    p_constant : float
        Constant probability for conditions
    p_specialist : float
        Specialist probability for new subtrees
        
    Returns
    -------
    Callable
        Mutation function
    """
    # Normalize probabilities
    total = p_specialist_mut + p_condition_mut + p_subtree_mut + p_hoist_mut + p_prune_mut
    if total <= 0:
        total = 1.0
    
    probs = [
        p_specialist_mut / total,
        p_condition_mut / total,
        p_subtree_mut / total,
        p_hoist_mut / total,
        p_prune_mut / total,
    ]
    cumprobs = np.cumsum(probs)
    
    def mutate(individual: EnsembleIndividual) -> EnsembleIndividual:
        r = rng.random()
        
        if r < cumprobs[0]:
            return mutate_specialist(individual, primitive_set, rng)
        elif r < cumprobs[1]:
            return mutate_condition(
                individual, primitive_set, rng,
                depth_condition, p_terminal, p_constant
            )
        elif r < cumprobs[2]:
            return mutate_subtree(
                individual, primitive_set, rng,
                depth_condition, max_depth, p_specialist, p_terminal, p_constant
            )
        elif r < cumprobs[3]:
            return mutate_hoist(individual, primitive_set, rng)
        else:
            return mutate_prune(individual, primitive_set, rng)
    
    return mutate


# =============================================================================
# Crossover Operators
# =============================================================================

def homologous_crossover(
    parent1: EnsembleIndividual,
    parent2: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 4,
) -> tuple[EnsembleIndividual, EnsembleIndividual]:
    """
    Perform homologous crossover between two ensemble individuals.
    
    Homologous crossover exchanges subtrees at matching positions
    in both trees, preserving structural similarity.
    
    Parameters
    ----------
    parent1 : EnsembleIndividual
        First parent
    parent2 : EnsembleIndividual
        Second parent
    primitive_set : PrimitiveSet
        Primitive set
    rng : np.random.Generator
        Random number generator
    max_depth : int
        Maximum allowed depth for offspring
        
    Returns
    -------
    tuple[EnsembleIndividual, EnsembleIndividual]
        Two offspring
    """
    # Find common branch positions
    branches1 = _get_all_branches(parent1.collection)
    branches2 = _get_all_branches(parent2.collection)
    
    # Find matching positions (same path exists in both trees)
    common_paths = []
    for path in branches1:
        if path in branches2:
            common_paths.append(path)
    
    if not common_paths:
        # No common positions, return copies
        return parent1.copy(), parent2.copy()
    
    # Choose random common position
    idx = rng.integers(0, len(common_paths))
    path = common_paths[idx]
    
    # Get subtrees at this position
    subtree1 = parent1.get_subtree(path)
    subtree2 = parent2.get_subtree(path)
    
    # Create offspring by swapping
    off1 = parent1.replace_subtree(path, _deep_copy_subtree(subtree2))
    off2 = parent2.replace_subtree(path, _deep_copy_subtree(subtree1))
    
    # Check depth constraints
    if off1.depth > max_depth:
        off1 = parent1.copy()
    if off2.depth > max_depth:
        off2 = parent2.copy()
    
    return off1, off2


def subtree_crossover(
    parent1: EnsembleIndividual,
    parent2: EnsembleIndividual,
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 4,
) -> tuple[EnsembleIndividual, EnsembleIndividual]:
    """
    Perform subtree crossover between two ensemble individuals.
    
    Exchanges random subtrees between parents.
    
    Parameters
    ----------
    parent1 : EnsembleIndividual
        First parent
    parent2 : EnsembleIndividual
        Second parent
    primitive_set : PrimitiveSet
        Primitive set
    rng : np.random.Generator
        Random number generator
    max_depth : int
        Maximum allowed depth for offspring
        
    Returns
    -------
    tuple[EnsembleIndividual, EnsembleIndividual]
        Two offspring
    """
    # Get all branch positions
    branches1 = _get_all_branches(parent1.collection)
    branches2 = _get_all_branches(parent2.collection)
    
    if not branches1 or not branches2:
        return parent1.copy(), parent2.copy()
    
    # Choose random positions
    idx1 = rng.integers(0, len(branches1))
    idx2 = rng.integers(0, len(branches2))
    path1 = branches1[idx1]
    path2 = branches2[idx2]
    
    # Get subtrees
    subtree1 = parent1.get_subtree(path1)
    subtree2 = parent2.get_subtree(path2)
    
    # Create offspring
    off1 = parent1.replace_subtree(path1, _deep_copy_subtree(subtree2))
    off2 = parent2.replace_subtree(path2, _deep_copy_subtree(subtree1))
    
    # Check depth constraints
    if off1.depth > max_depth:
        off1 = parent1.copy()
    if off2.depth > max_depth:
        off2 = parent2.copy()
    
    return off1, off2


def _get_all_branches(tree: EnsembleTreeRepr, path: Optional[list] = None) -> list[list[int]]:
    """Get all valid branch paths in the tree."""
    if path is None:
        path = []
    
    branches = []
    
    if isinstance(tree, tuple):
        # Add paths to true branch (1) and false branch (2)
        branches.append(path + [1])
        branches.append(path + [2])
        
        # Recurse
        branches.extend(_get_all_branches(tree[1], path + [1]))
        branches.extend(_get_all_branches(tree[2], path + [2]))
    
    return branches


# =============================================================================
# Factory Functions
# =============================================================================

def fetch_ensemble_crossover(
    crossover_type: str = "homologous",
    max_depth: int = 4,
) -> Callable:
    """
    Factory function to create ensemble crossover operators.
    
    Parameters
    ----------
    crossover_type : str
        Type of crossover: "homologous" or "subtree"
    max_depth : int
        Maximum depth for offspring
        
    Returns
    -------
    Callable
        Crossover function
    """
    if crossover_type == "homologous":
        def crossover(p1, p2, pset, rng, max_depth=max_depth):
            return homologous_crossover(p1, p2, pset, rng, max_depth)
        return crossover
    elif crossover_type == "subtree":
        def crossover(p1, p2, pset, rng, max_depth=max_depth):
            return subtree_crossover(p1, p2, pset, rng, max_depth)
        return crossover
    else:
        raise ValueError(f"Unknown crossover type: {crossover_type}")


def fetch_ensemble_mutation(
    primitive_set: PrimitiveSet,
    rng: np.random.Generator,
    depth_condition: int = 3,
    max_depth: int = 4,
    p_specialist_mut: float = 0.3,
    p_condition_mut: float = 0.3,
    p_subtree_mut: float = 0.2,
    p_hoist_mut: float = 0.1,
    p_prune_mut: float = 0.1,
) -> Callable:
    """
    Factory function to create ensemble mutation operators.
    
    Parameters
    ----------
    primitive_set : PrimitiveSet
        Primitive set with specialists
    rng : np.random.Generator
        Random number generator
    depth_condition : int
        Maximum depth for condition trees
    max_depth : int
        Maximum depth for ensemble trees
    p_specialist_mut : float
        Probability of specialist swap mutation
    p_condition_mut : float
        Probability of condition mutation
    p_subtree_mut : float
        Probability of subtree mutation
    p_hoist_mut : float
        Probability of hoist mutation
    p_prune_mut : float
        Probability of prune mutation
        
    Returns
    -------
    Callable
        Mutation function
    """
    return ensemble_mutator(
        primitive_set=primitive_set,
        rng=rng,
        depth_condition=depth_condition,
        max_depth=max_depth,
        p_specialist_mut=p_specialist_mut,
        p_condition_mut=p_condition_mut,
        p_subtree_mut=p_subtree_mut,
        p_hoist_mut=p_hoist_mut,
        p_prune_mut=p_prune_mut,
    )


# =============================================================================
# Variator Function
# =============================================================================

def ensemble_variator(
    crossover: Callable,
    mutation: Callable,
    p_xo: float = 0.5,
) -> Callable:
    """
    Create a variator function that applies crossover or mutation.
    
    Compatible with GPevo: accepts Population and returns Population.
    
    Parameters
    ----------
    crossover : Callable
        Crossover function
    mutation : Callable
        Mutation function
    p_xo : float
        Probability of crossover (vs mutation)
        
    Returns
    -------
    Callable
        Variator function that accepts (Population, PrimitiveSet, rng, max_depth)
        and returns (Population, timing_dict)
    """
    def variator(
        parents: Population,
        primitive_set: PrimitiveSet,
        rng: np.random.Generator,
        max_depth: int,
    ) -> tuple[Population, dict]:
        """Apply variation to create offspring population."""
        import time
        
        # Extract individuals list from Population
        parent_list = parents.population
        
        offspring = []
        xo_time = 0.0
        mut_time = 0.0
        
        pop_size = len(parent_list)
        i = 0
        
        while len(offspring) < pop_size:
            if rng.random() < p_xo and i + 1 < pop_size:
                # Crossover
                start = time.perf_counter()
                off1, off2 = crossover(
                    parent_list[i], parent_list[i + 1],
                    primitive_set, rng, max_depth
                )
                xo_time += time.perf_counter() - start
                
                offspring.extend([off1, off2])
                i += 2
            else:
                # Mutation
                start = time.perf_counter()
                off = mutation(parent_list[i % pop_size])
                mut_time += time.perf_counter() - start
                
                offspring.append(off)
                i += 1
        
        # Trim to exact size
        offspring = offspring[:pop_size]
        
        timing_info = {
            "xo_time": xo_time,
            "mut_time": mut_time,
        }
        
        return Population(offspring), timing_info
    
    return variator

