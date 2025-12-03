import numpy as np
from sklearn.model_selection import train_test_split

from pssr import GPRegressor
from datasets.data_loader import load_airfoil
from sklearn.metrics import r2_score

def test_gp_fit_and_predict_shapes():
    X, y = load_airfoil()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gp = GPRegressor(
        population_size=100,
        max_depth=7,
        random_state=0,
        selector="dalex",
        selector_args={
            "particularity_pressure" : 10.0
        },
    )
    
    gp.fit(X_train, y_train, n_gen=200, verbose=1)
    y_pred = gp.predict(X_test)
    
    # calculate r2 
    r2 = r2_score(y_test, y_pred)
    print(f"R2 score: {r2}")
    
    assert y_pred.shape == y_test.shape


def test_gp_reproducible_with_random_state():
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.sin(X[:, 0])

    gp1 = GPRegressor(population_size=30, random_state=123)
    gp2 = GPRegressor(population_size=30, random_state=123)

    gp1.fit(X, y, n_gen=10)
    gp2.fit(X, y, n_gen=10)

    y1 = gp1.predict(X)
    y2 = gp2.predict(X)

    assert np.allclose(y1, y2)


def test_gp_warm_start():
    """Test that warm start allows continuing training."""
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.sin(X[:, 0])

    gp = GPRegressor(population_size=100, random_state=123)
    
    # First fit
    gp.fit(X, y, n_gen=20, verbose=1)
    initial_best = gp.best_individual_.fitness
    initial_gen_count = len(gp.log_["generation"])
    
    # Warm start: continue training
    gp.fit(X, y, n_gen=20, verbose=1, warm_start=True)
    final_best = gp.best_individual_.fitness
    final_gen_count = len(gp.log_["generation"])
    
    # Check that we continued from where we left off
    # First fit: 20 generations (0-20), second fit: 20 more generations (20-40)
    assert final_gen_count == initial_gen_count + 20, \
        f"Expected {initial_gen_count + 20} generations, got {final_gen_count}"
    
    # Check that log generations are continuous
    assert gp.log_["generation"][-1] == initial_gen_count + 19, \
        f"Last generation should be {initial_gen_count + 19}, got {gp.log_['generation'][-1]}"
    
    print(f"Warm start test passed: {initial_gen_count} -> {final_gen_count} generations")


def test_gp_test_evaluation():
    """Test that test evaluation works during fitting."""
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = np.sin(X[:, 0])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gp = GPRegressor(population_size=50, random_state=123)
    
    # Fit with test data
    gp.fit(X_train, y_train, X_test=X_test, y_test=y_test, n_gen=100, verbose=1)
    
    # Check that test fitness is in log
    assert "best_test_fitness" in gp.log_, "Test fitness should be in log"
    assert len(gp.log_["best_test_fitness"]) == len(gp.log_["generation"]), \
        "Test fitness should have same length as generation log"
    
    # Check that test fitness values are reasonable (non-negative)
    assert all(f >= 0 for f in gp.log_["best_test_fitness"]), \
        "Test fitness should be non-negative"
    
    # Check that test column appears in verbose output (tested by checking log structure)
    print("Test evaluation test passed: test fitness tracked in log")


if __name__ == "__main__":
    test_gp_fit_and_predict_shapes()
    test_gp_reproducible_with_random_state()
    test_gp_warm_start()
    test_gp_test_evaluation()
    print("All tests passed")