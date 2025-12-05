"""
PSSR Model: Piecewise Specialist Symbolic Regression.

This module implements the Multi-SLIM approach for symbolic regression
using piecewise specialists with conditional routing.
"""

from pssr.pssr_model.condition import Condition
from pssr.pssr_model.ensemble_individual import EnsembleIndividual
from pssr.pssr_model.ensemble_initialization import (
    create_grow_random_condition,
    create_random_ensemble_tree,
    ensemble_initializer,
    fetch_ensemble_initializer,
)
from pssr.pssr_model.ensemble_operators import (
    ensemble_variator,
    fetch_ensemble_crossover,
    fetch_ensemble_mutation,
    homologous_crossover,
    mutate_condition,
    mutate_hoist,
    mutate_prune,
    mutate_specialist,
    mutate_subtree,
    subtree_crossover,
)
from pssr.pssr_model.pss_regressor import PSSRegressor
from pssr.pssr_model.specialist import Specialist, create_specialists_from_population

__all__ = [
    # Main regressor
    "PSSRegressor",
    
    # Core components
    "Specialist",
    "Condition",
    "EnsembleIndividual",
    
    # Initialization
    "ensemble_initializer",
    "fetch_ensemble_initializer",
    "create_random_ensemble_tree",
    "create_grow_random_condition",
    "create_specialists_from_population",
    
    # Operators
    "fetch_ensemble_crossover",
    "fetch_ensemble_mutation",
    "ensemble_variator",
    "mutate_specialist",
    "mutate_condition",
    "mutate_subtree",
    "mutate_hoist",
    "mutate_prune",
    "homologous_crossover",
    "subtree_crossover",
]

