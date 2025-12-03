"""Profile GPRegressor to identify performance bottlenecks."""

import cProfile
import pstats
import io
from pathlib import Path
import time

import numpy as np

from pssr import GPRegressor
from pssr.core.representations.population import Population


def create_test_data(n_samples=200, n_features=2, seed=42):
    """Create test data for profiling."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, n_features))
    # Create a simple target function
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    return X, y


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Simple train/test split without sklearn dependency."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def profile_gp_regressor():
    """Profile GPRegressor fit and predict operations with train/test split."""
    print("=" * 80)
    print("GPRegressor Profiling (with Train/Test Split)")
    print("=" * 80)
    
    # Create test data
    X, y = create_test_data(n_samples=1000, n_features=6, seed=42)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, seed=42)
    
    print(f"\nDataset split:")
    print(f"  Total data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Train data: {X_train.shape[0]} samples")
    print(f"  Test data: {X_test.shape[0]} samples")
    
    # Create GPRegressor with moderate settings for profiling
    gp = GPRegressor(
        population_size=100,
        max_depth=6,
        random_state=42,
        selector="dalex",
        p_xo=0.8,
        selector_args={"particularity_pressure": 10.0},
    )
    
    print(f"\nGPRegressor settings:")
    print(f"  Population size: {gp.population_size}")
    print(f"  Max depth: {gp.max_depth}")
    
    # Profile the fit operation WITH test data
    print("\n" + "=" * 80)
    print("Profiling fit() operation WITH test data...")
    print("=" * 80)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    gp.fit(X_train, y_train, X_test=X_test, y_test=y_test, n_gen=500, verbose=1)
    
    profiler.disable()
    
    # Get profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print(s.getvalue())
    
    # Save detailed profile to file
    profile_file = Path("gp_regressor_profile.prof")
    profiler.dump_stats(str(profile_file))
    print(f"\nDetailed profile saved to: {profile_file}")
    print("View with: python -m pstats gp_regressor_profile.prof")
    
    # Profile predict operation on test set
    print("\n" + "=" * 80)
    print("Profiling predict() operation on test set...")
    print("=" * 80)
    
    profiler2 = cProfile.Profile()
    profiler2.enable()
    
    y_pred = gp.predict(X_test)
    
    profiler2.disable()
    
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler2, stream=s2)
    ps2.sort_stats('cumulative')
    ps2.print_stats(20)  # Top 20 functions
    
    print(s2.getvalue())
    
    # Time breakdown analysis
    print("\n" + "=" * 80)
    print("Time Breakdown Analysis")
    print("=" * 80)
    
    # Analyze the profile stats
    stats = pstats.Stats(profiler)
    
    # Group by module
    module_times = {}
    for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
        module = func_name[0]  # filename
        if module not in module_times:
            module_times[module] = {'cumulative': 0, 'total': 0, 'calls': 0}
        module_times[module]['cumulative'] += ct
        module_times[module]['total'] += tt
        module_times[module]['calls'] += nc
    
    # Sort by cumulative time
    sorted_modules = sorted(
        module_times.items(),
        key=lambda x: x[1]['cumulative'],
        reverse=True
    )
    
    print("\nTop modules by cumulative time:")
    print(f"{'Module':<60} {'Cumulative Time':<20} {'Total Time':<20} {'Calls':<10}")
    print("-" * 110)
    for module, times in sorted_modules[:15]:
        print(
            f"{module[:58]:<60} "
            f"{times['cumulative']:<20.4f} "
            f"{times['total']:<20.4f} "
            f"{times['calls']:<10}"
        )
    
    # Find slowest individual functions
    print("\n" + "=" * 80)
    print("Top 20 Slowest Functions (by cumulative time)")
    print("=" * 80)
    
    func_times = []
    for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
        func_times.append((func_name, ct, tt, nc))
    
    func_times.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Function':<70} {'Cumulative':<15} {'Total':<15} {'Calls':<10}")
    print("-" * 110)
    for func_name, ct, tt, nc in func_times[:20]:
        func_str = f"{func_name[2]}.{func_name[1]}" if len(func_name) > 2 else str(func_name[1])
        print(f"{func_str[:68]:<70} {ct:<15.4f} {tt:<15.4f} {nc:<10}")
    
    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("=" * 80)
    
    return gp, y_pred, X_test, y_test


def benchmark_semantics_evaluation():
    """Compare evaluation time between combined vs separate semantics computation."""
    print("\n" + "=" * 80)
    print("Benchmark: Combined vs Separate Semantics Evaluation")
    print("=" * 80)
    
    X, y = create_test_data(n_samples=800, n_features=4, seed=7)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.25, seed=7)
    
    # Build combined dataset for semantics
    X_semantics = np.vstack([X_train, X_test])
    train_slice = slice(0, X_train.shape[0])
    test_slice = slice(X_train.shape[0], X_semantics.shape[0])
    
    gp = GPRegressor(population_size=80, max_depth=5, random_state=123)
    gp._resolve_preset()
    gp._resolve_primitive_set(X_semantics)
    gp._rng()
    gp._resolve_selector()
    gp._resolve_variator()
    initializer = gp._resolve_initializer()
    individuals = initializer(
        gp.primitive_set_,
        gp._rng_,
        gp.population_size,
        gp.max_depth,
    )
    population = Population(individuals)
    population_copy = population.copy()
    
    # Combined semantics
    start = time.perf_counter()
    population.calculate_semantics_combined(X_semantics, train_slice=train_slice, test_slice=test_slice)
    combined_time = time.perf_counter() - start
    
    # Separate semantics (train then test)
    start = time.perf_counter()
    population_copy.calculate_semantics(X_train, testing=False)
    population_copy.calculate_semantics(X_test, testing=True)
    separate_time = time.perf_counter() - start
    
    improvement = (separate_time - combined_time) / separate_time * 100 if separate_time > 0 else 0
    
    print(f"Combined evaluation time : {combined_time:.4f} s")
    print(f"Separate evaluation time : {separate_time:.4f} s")
    print(f"Improvement              : {improvement:.2f}% faster")


if __name__ == "__main__":
    gp, y_pred, X_test, y_test = profile_gp_regressor()
    
    # Print prediction accuracy
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nFinal Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    
    benchmark_semantics_evaluation()

