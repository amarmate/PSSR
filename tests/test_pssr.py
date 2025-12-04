import numpy as np

from pssr import PSSRegressor
from pssr.core.representations.individual import Individual
from pssr.core.primitives import PrimitiveSet
from pssr.pssr_model.ensemble_individual import EnsembleIndividual
from pssr.pssr_model.specialist import Specialist


def _generate_piecewise_linear(n=200, noise=0.0, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2, 2, size=(n, 1))
    y = np.where(
        X[:, 0] < 0,
        2 * X[:, 0] + 1,      # left region
        -X[:, 0] + 2          # right region
    )
    y += noise * rng.normal(size=n)
    return X, y


def test_pssr_fit_predict():
    X, y = _generate_piecewise_linear(n=100, noise=0.1)

    model = PSSRegressor(
        specialist_pop_size=20,
        ensemble_pop_size=20,
        random_state=0,
    )
    model.fit(X, y, specialist_n_gen=5, ensemble_n_gen=5)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape


def test_pssr_with_test_data():
    """Test PSSR with train/test split."""
    X, y = _generate_piecewise_linear(n=100, noise=0.1)
    
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    model = PSSRegressor(
        specialist_pop_size=500,
        ensemble_pop_size=100,
        random_state=0,
        specialist_selector="dalex",
        ensemble_selector="tournament",
        specialist_selector_args={"particularity_pressure": 20},
        ensemble_selector_args={"pool_size": 2},
        fitness_function="r2",
    )
    model.fit(
        X_train, y_train,
        X_test=X_test, y_test=y_test,
        specialist_n_gen=100, ensemble_n_gen=500,
        verbose=50,
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    assert y_pred_train.shape == y_train.shape
    assert y_pred_test.shape == y_test.shape


def test_pssr_sklearn_compatibility():
    """Test sklearn RegressorMixin compatibility (score method)."""
    X, y = _generate_piecewise_linear(n=100, noise=0.1)
    
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    model = PSSRegressor(
        specialist_pop_size=15,
        ensemble_pop_size=15,
        random_state=0,
    )
    model.fit(X_train, y_train, specialist_n_gen=3, ensemble_n_gen=3)
    
    # Test sklearn score method (R^2)
    score = model.score(X_test, y_test)
    assert isinstance(score, float)
    # Should be reasonable
    assert score > -10.0


def test_individual_vs_ensemble_individual_no_conditions():
    """
    Test that an EnsembleIndividual with no conditions (just a single specialist)
    produces the same semantics and fitness as the underlying Individual.
    
    Steps:
    1. Create an Individual
    2. Create an EnsembleIndividual that doesn't have conditions and only has that individual
    3. Test if their semantics outputs are the same, fitness, etc.
    """
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(100)
    
    # Step 1: Create an Individual
    pset_individual = PrimitiveSet(X, functions=['add', 'sub', 'mul'], constant_range=1.0)
    tree = ('add', ('mul', 'x0', 'x1'), 'x0')
    individual = Individual(tree, primitive_set=pset_individual)
    
    # Calculate semantics and fitness for the individual
    individual.set_slices(slice(0, 100))
    individual_semantics = individual.calculate_semantics(X)
    individual_fitness = individual.evaluate(y)
    
    # Step 2: Create a Specialist from the Individual
    specialist = Specialist(name="S_0", individual=individual)
    specialist.compute_semantics(X)
    
    # Create an EnsembleIndividual that just uses the specialist (no conditions)
    # The tree is just the specialist name (a string, not a tuple)
    ensemble_tree = "S_0"
    
    # Create primitive set with the specialist
    pset_ensemble = PrimitiveSet(X, functions=['add', 'sub', 'mul'], constant_range=1.0)
    pset_ensemble.set_specialists({"S_0": specialist})
    
    ensemble_individual = EnsembleIndividual(
        collection=ensemble_tree,
        primitive_set=pset_ensemble
    )
    
    # Calculate semantics and fitness for the ensemble individual
    ensemble_individual.set_slices(slice(0, 100))
    ensemble_semantics = ensemble_individual.calculate_semantics(X)
    ensemble_fitness = ensemble_individual.evaluate(y)
    
    # Step 3: Test if their semantics outputs are the same
    np.testing.assert_array_almost_equal(
        individual_semantics,
        ensemble_semantics,
        decimal=10,
        err_msg="Individual and EnsembleIndividual semantics should be identical"
    )
    
    # Test if their fitness is the same
    np.testing.assert_almost_equal(
        individual_fitness,
        ensemble_fitness,
        decimal=10,
        err_msg="Individual and EnsembleIndividual fitness should be identical"
    )
    
    # Test if train_semantics are the same
    np.testing.assert_array_almost_equal(
        individual.train_semantics,
        ensemble_individual.train_semantics,
        decimal=10,
        err_msg="Individual and EnsembleIndividual train_semantics should be identical"
    )
    
    # Test predictions on new data
    X_new = np.random.randn(50, 2)
    individual_pred = individual.predict(X_new)
    ensemble_pred = ensemble_individual.predict(X_new, primitive_set=pset_ensemble)
    
    np.testing.assert_array_almost_equal(
        individual_pred,
        ensemble_pred,
        decimal=10,
        err_msg="Individual and EnsembleIndividual predictions should be identical"
    )
    
    print("✓ All tests passed: Individual and EnsembleIndividual produce identical results")


if __name__ == "__main__":
    test_pssr_fit_predict()
    test_pssr_with_test_data()
    test_pssr_sklearn_compatibility()
    test_individual_vs_ensemble_individual_no_conditions()
