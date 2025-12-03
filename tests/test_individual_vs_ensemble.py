"""
Test to verify that an EnsembleIndividual with a single specialist (no conditions)
produces the same semantics and fitness as the underlying Individual.
"""

import numpy as np
import pytest

from pssr.core.primitives import PrimitiveSet
from pssr.core.representations.individual import Individual
from pssr.pssr_model.ensemble_individual import EnsembleIndividual
from pssr.pssr_model.specialist import Specialist
from pssr.gp.gp_initialization import grow_initializer


def test_individual_vs_ensemble_individual():
    """
    Test that an EnsembleIndividual with a single specialist (no conditions)
    produces identical results to the underlying Individual.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    # Create test data
    n_samples = 100
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # Split into train and test
    n_train = 70
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    # Create combined data
    X_combined = np.vstack([X_train, X_test])
    train_slice = slice(0, n_train)
    test_slice = slice(n_train, n_samples)
    
    # =====================================================================
    # Step 1: Create an Individual
    # =====================================================================
    pset = PrimitiveSet(X_combined, functions=["add", "sub", "mul"], constant_range=1.0)
    
    # Create a simple individual using grow initializer
    individuals = grow_initializer(
        primitive_set=pset,
        rng=rng,
        population_size=1,
        max_depth=3,
        init_depth=2,
    )
    individual = individuals[0]
    
    # Calculate semantics for the individual
    individual.set_slices(train_slice, test_slice)
    individual_semantics = individual.calculate_semantics(X_combined, primitive_set=pset)
    
    # Evaluate fitness
    individual.evaluate(y_train, y_test=y_test)
    
    print(f"\nIndividual:")
    print(f"  Tree: {individual.tree}")
    print(f"  Depth: {individual.depth}")
    print(f"  Total nodes: {individual.total_nodes}")
    print(f"  Fitness: {individual.fitness:.6f}")
    print(f"  Test fitness: {individual.test_fitness:.6f}")
    print(f"  Train semantics shape: {individual.train_semantics.shape}")
    print(f"  Test semantics shape: {individual.test_semantics.shape}")
    
    # =====================================================================
    # Step 2: Create a Specialist from the Individual
    # =====================================================================
    specialist = Specialist(
        name="S_0",
        individual=individual,
    )
    specialist.compute_semantics(X_combined)
    
    print(f"\nSpecialist:")
    print(f"  Name: {specialist.name}")
    print(f"  Semantics shape: {specialist.semantics.shape}")
    print(f"  Fitness: {specialist.fitness:.6f}")
    print(f"  Test fitness: {specialist.test_fitness:.6f}")
    
    # =====================================================================
    # Step 3: Create EnsembleIndividual with only the specialist (no conditions)
    # =====================================================================
    # Create ensemble primitive set with the specialist
    ensemble_pset = PrimitiveSet(X_combined, functions=["add", "sub", "mul"], constant_range=1.0)
    ensemble_pset.set_specialists({"S_0": specialist})
    
    # Create ensemble individual with just the specialist (no conditions)
    # The tree is just "S_0" - a single specialist terminal
    ensemble_tree = "S_0"
    ensemble_individual = EnsembleIndividual(
        collection=ensemble_tree,
        primitive_set=ensemble_pset,
    )
    
    # Calculate semantics
    ensemble_individual.set_slices(train_slice, test_slice)
    ensemble_semantics = ensemble_individual.calculate_semantics(X_combined, primitive_set=ensemble_pset)
    
    # Evaluate fitness
    ensemble_individual.evaluate(y_train, y_test=y_test)
    
    print(f"\nEnsembleIndividual:")
    print(f"  Tree: {ensemble_individual.collection}")
    print(f"  Depth: {ensemble_individual.depth}")
    print(f"  Total nodes: {ensemble_individual.total_nodes}")
    print(f"  Fitness: {ensemble_individual.fitness:.6f}")
    print(f"  Test fitness: {ensemble_individual.test_fitness:.6f}")
    print(f"  Train semantics shape: {ensemble_individual.train_semantics.shape}")
    print(f"  Test semantics shape: {ensemble_individual.test_semantics.shape}")
    
    # =====================================================================
    # Step 4: Compare results
    # =====================================================================
    print(f"\nComparison:")
    
    # Compare combined semantics
    np.testing.assert_allclose(
        individual_semantics,
        ensemble_semantics,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Combined semantics should be identical"
    )
    print("  ✓ Combined semantics match")
    
    # Compare train semantics
    np.testing.assert_allclose(
        individual.train_semantics,
        ensemble_individual.train_semantics,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Train semantics should be identical"
    )
    print("  ✓ Train semantics match")
    
    # Compare test semantics
    np.testing.assert_allclose(
        individual.test_semantics,
        ensemble_individual.test_semantics,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Test semantics should be identical"
    )
    print("  ✓ Test semantics match")
    
    # Compare fitness
    assert abs(individual.fitness - ensemble_individual.fitness) < 1e-10, \
        f"Fitness should be identical: {individual.fitness} vs {ensemble_individual.fitness}"
    print(f"  ✓ Fitness match: {individual.fitness:.10f}")
    
    # Compare test fitness
    assert abs(individual.test_fitness - ensemble_individual.test_fitness) < 1e-10, \
        f"Test fitness should be identical: {individual.test_fitness} vs {ensemble_individual.test_fitness}"
    print(f"  ✓ Test fitness match: {individual.test_fitness:.10f}")
    
    # Compare predictions on new data
    X_new = np.random.randn(20, n_features)
    individual_pred = individual.predict(X_new, primitive_set=pset)
    ensemble_pred = ensemble_individual.predict(X_new, primitive_set=ensemble_pset)
    
    np.testing.assert_allclose(
        individual_pred,
        ensemble_pred,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Predictions on new data should be identical"
    )
    print("  ✓ Predictions on new data match")
    
    print("\n✅ All tests passed! Individual and EnsembleIndividual produce identical results.")


if __name__ == "__main__":
    test_individual_vs_ensemble_individual()

