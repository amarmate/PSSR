"""Profile old GPRegressor from slim_gsgp_lib_np to identify performance bottlenecks."""

import cProfile
import pstats
import io
from pathlib import Path

import numpy as np

from slim_gsgp_lib_np.main_gp import gp


def create_test_data(n_samples=200, n_features=2, seed=42):
    """Create test data for profiling."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, n_features))
    # Create a simple target function
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    return X, y


def profile_old_gp_regressor():
    """Profile old GPRegressor fit and predict operations."""
    print("=" * 80)
    print("Old GPRegressor (slim_gsgp_lib_np) Profiling")
    print("=" * 80)
    
    # Create test data
    X, y = create_test_data(n_samples=1000, n_features=6, seed=42)
    print(f"\nTest data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split into train/test (old GP requires separate test set)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    
    # Create GPRegressor with moderate settings for profiling
    # Matching the settings from profile_gp_regressor.py
    print(f"\nOld GPRegressor settings:")
    print(f"  Population size: 100")
    print(f"  Generations: 500")
    print(f"  Max depth: 6")
    print(f"  Selector: dalex")
    print(f"  Particularity pressure: 10.0")
    
    # Profile the fit operation (solve method)
    print("\n" + "=" * 80)
    print("Profiling solve() operation...")
    print("=" * 80)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the old GP algorithm
    optimizer = gp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="profile_test",
        pop_size=100,
        n_iter=500,
        max_depth=6,
        selector="dalex",
        particularity_pressure=10.0,
        p_xo=0.8,
        seed=42,
        verbose=1,
        test_elite=False,  # Set to False to speed up profiling
        full_return=True,  # Return optimizer to access elite
    )
    
    profiler.disable()
    
    # Get profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print(s.getvalue())
    
    # Save detailed profile to file
    profile_file = Path("old_gp_regressor_profile.prof")
    profiler.dump_stats(str(profile_file))
    print(f"\nDetailed profile saved to: {profile_file}")
    print("View with: python -m pstats old_gp_regressor_profile.prof")
    
    # Profile predict operation
    print("\n" + "=" * 80)
    print("Profiling predict() operation...")
    print("=" * 80)
    
    profiler2 = cProfile.Profile()
    profiler2.enable()
    
    y_pred = optimizer.elite.predict(X_test)
    
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
    
    return optimizer, y_pred


if __name__ == "__main__":
    optimizer, y_pred = profile_old_gp_regressor()

