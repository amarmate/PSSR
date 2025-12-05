import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ensure old library is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slim_gsgp_lib_np.datasets import data_loader as old_data
from slim_gsgp_lib_np.main_multi_slim import multi_slim

from pssr.pssr_model.pss_regressor import PSSRegressor


# ------------------------------
# Parameter configurations
# ------------------------------
CONSTANTS_LIST = [round(i * 0.05, 2) for i in range(2, 21)] + [round(-i * 0.05, 2) for i in range(2, 21)]
CONSTANT_RANGE = max(abs(c) for c in CONSTANTS_LIST)
FUNCTIONS_LIST = ['add', 'subtract', 'multiply', 'divide']

CONFIG_LARGE: Dict[str, Any] = {
    "name": "config_large",
    
    # Specialist phase
    "specialist_pop_size": 200,
    "specialist_n_iter": 1000,
    "specialist_max_depth": 6,
    "specialist_init_depth": 2,
    
    # Ensemble phase
    "ensemble_pop_size": 100,
    "ensemble_n_iter": 2000,
    "ensemble_max_depth": 5,
    "depth_condition": 4,
    
    # Shared variation / selection
    "p_xo": 0.8,
    "selector": "dalex",
    "particularity_pressure": 20.0,
    "fitness_function": "rmse",
    "n_elites": 1,
    "seed": 44,
    
    # Constants / functions
    "prob_const": 0.2,
    "prob_terminal": 0.7,
    "functions": FUNCTIONS_LIST,
    "constants": CONSTANTS_LIST,
    "constant_range": CONSTANT_RANGE,
}

# Single configuration list retained for minimal downstream changes
CONFIGS: List[Dict[str, Any]] = [CONFIG_LARGE]

# Number of samples to run per dataset
SAMPLE_SIZE = 30


# ------------------------------
# Dataset registry (small subset)
# ------------------------------
DATASETS = {
    "airfoil": old_data.load_airfoil,
    # "boston": old_data.load_boston,
    # "concrete_strength": old_data.load_concrete_strength,
    # "diabetes": old_data.load_diabetes,
    # "efficiency_heating": old_data.load_efficiency_heating,
}


# ------------------------------
# Helpers
# ------------------------------
def map_function_names_to_pssr(function_names: List[str]) -> List[str]:
    """Map MULTI_SLIM function names to PSSRegressor function names."""
    name_mapping = {
        "add": "add",
        "subtract": "sub",
        "multiply": "mul",
        "divide": "div",
        "sqrt": "sqrt",
    }
    mapped = []
    for name in function_names:
        if name in name_mapping:
            mapped.append(name_mapping[name])
        # Skip "sq" as PSSR doesn't have a direct equivalent (sq is x^2, but PSSR uses pow which requires 2 args)
    return mapped


def map_params_to_pssr(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map old MULTI_SLIM-style params to PSSRegressor params."""
    selector_args = {
        "particularity_pressure": cfg["particularity_pressure"],
    }
    # Map function names from MULTI_SLIM format to PSSRegressor format
    pssr_functions = map_function_names_to_pssr(cfg["functions"])
    return {
        "specialist_pop_size": cfg["specialist_pop_size"],
        "specialist_n_elites": cfg["n_elites"],
        "specialist_max_depth": cfg["specialist_max_depth"],
        "specialist_init_depth": cfg.get("specialist_init_depth", 2),
        "specialist_selector": cfg["selector"],
        "specialist_selector_params": selector_args,
        "specialist_initializer_params": {
            "p_c": cfg["prob_const"],
            "p_t": cfg["prob_terminal"],
        },
        "specialist_crossover": "single_point",
        "specialist_mutation": "subtree",
        "specialist_mutation_params": {},
        "ensemble_pop_size": cfg["ensemble_pop_size"],
        "ensemble_n_elites": cfg["n_elites"],
        "ensemble_max_depth": cfg["ensemble_max_depth"],
        "depth_condition": cfg["depth_condition"],
        "ensemble_selector": "tournament",
        "p_xo": cfg["p_xo"],
        "random_state": cfg["seed"],
        "fitness_function": cfg["fitness_function"],
        "functions": pssr_functions,
        "constant_range": cfg["constant_range"],
    }


def map_params_to_old_gp(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map params for the old GP (specialist) phase inside MULTI_SLIM.
    
    Note: Parameters that multi_slim passes explicitly (seed, verbose, log_path,
    test_elite, full_return, run_info, minimization) should NOT be included here
    to avoid duplicate keyword arguments.
    """
    return {
        "pop_size": cfg["specialist_pop_size"],
        "n_iter": cfg["specialist_n_iter"],
        "p_xo": cfg["p_xo"],
        "elitism": True,
        "n_elites": cfg["n_elites"],
        "selector": cfg["selector"],
        "std_errs": True,
        "max_depth": cfg["specialist_max_depth"],
        "init_depth": cfg.get("specialist_init_depth", 2),
        "log_level": 0,
        "fitness_function": cfg["fitness_function"],
        "initializer": "rhh",
        "n_jobs": 1,
        "prob_const": cfg["prob_const"],
        "prob_terminal": cfg["prob_terminal"],
        "prob_cond": 0.0,
        "tree_functions": cfg["functions"],
        "tree_constants": cfg["constants"],
        "particularity_pressure": cfg["particularity_pressure"],
        "callbacks": None,
        "elite_tree": None,
        "it_tolerance": 1,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_pssr(X_train, X_test, y_train, y_test, cfg: Dict[str, Any]) -> Dict[str, Any]:
    pss_params = map_params_to_pssr(cfg)
    model = PSSRegressor(**pss_params)
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        specialist_n_gen=cfg["specialist_n_iter"],
        ensemble_n_gen=cfg["ensemble_n_iter"],
        verbose=100,
    )
    elapsed = time.perf_counter() - start
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    # Capture ensemble size
    best_ensemble = model.get_best_ensemble()
    ensemble_nodes = getattr(best_ensemble, "nodes_count", None)
    ensemble_total_nodes = getattr(best_ensemble, "total_nodes", ensemble_nodes)
    
    return {
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "time_sec": elapsed,
        "generations": cfg["specialist_n_iter"] + cfg["ensemble_n_iter"],
        "specialist_generations": cfg["specialist_n_iter"],
        "ensemble_generations": cfg["ensemble_n_iter"],
        "ensemble_nodes_count": ensemble_nodes,
        "ensemble_total_nodes": ensemble_total_nodes,
    }


def run_multi_slim(X_train, X_test, y_train, y_test, cfg: Dict[str, Any]) -> Dict[str, Any]:
    params_gp = map_params_to_old_gp(cfg)
    start = time.perf_counter()
    elite = multi_slim(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="dataset",
        params_gp=params_gp,
        gp_version="gp",
        pop_size=cfg["ensemble_pop_size"],
        n_iter=cfg["ensemble_n_iter"],
        p_xo=cfg["p_xo"],
        depth_condition=cfg["depth_condition"],
        max_depth=cfg["ensemble_max_depth"],
        prob_const=cfg["prob_const"],
        prob_terminal=cfg["prob_terminal"],
        prob_specialist=0.7,
        test_elite=False,
        n_elites=cfg["n_elites"],
        fitness_function=cfg["fitness_function"],
        seed=cfg["seed"],
        verbose=0,
        log_level=0,
        minimization=True,
        log_path=None,
        selector=cfg["selector"],
        std_errs=True,
        particularity_pressure=cfg["particularity_pressure"],
        decay_rate=cfg.get("decay_rate", 0.1),
        ensemble_functions=cfg["functions"],
        ensemble_constants=cfg["constants"],
        callbacks=None,
        timeout=999999,  # Large value to effectively disable timeout
        full_return=False,
        elite_tree=None,
        it_tolerance=1,
    )
    elapsed = time.perf_counter() - start
    y_pred_train = elite.predict(X_train)
    y_pred_test = elite.predict(X_test)
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    elite_nodes = getattr(elite, "nodes_count", None)
    elite_total_nodes = getattr(elite, "total_nodes", elite_nodes)
    
    return {
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "time_sec": elapsed,
        "generations": cfg["specialist_n_iter"] + cfg["ensemble_n_iter"],
        "specialist_generations": cfg["specialist_n_iter"],
        "ensemble_generations": cfg["ensemble_n_iter"],
        "ensemble_nodes_count": elite_nodes,
        "ensemble_total_nodes": elite_total_nodes,
    }


def _paired_metric_values(
    pssr_results: List[Dict[str, Any]],
    ms_results: List[Dict[str, Any]],
    metric: str,
    dataset: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect paired metric arrays (PSSR, MULTI_SLIM) for a metric and optional dataset."""
    p_map = {
        (r["dataset"], r["sample_id"]): r[metric]
        for r in pssr_results
        if dataset is None or r["dataset"] == dataset
    }
    m_map = {
        (r["dataset"], r["sample_id"]): r[metric]
        for r in ms_results
        if dataset is None or r["dataset"] == dataset
    }
    keys = sorted(set(p_map.keys()) & set(m_map.keys()))
    p_vals = np.array([p_map[k] for k in keys], dtype=float)
    m_vals = np.array([m_map[k] for k in keys], dtype=float)
    return p_vals, m_vals


def compute_statistical_tests(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run paired t-test and Wilcoxon signed-rank tests between PSSR and MULTI_SLIM."""
    metrics = [
        "test_rmse",
        "test_mae",
        "test_r2",
        "time_sec",
        "ensemble_nodes_count",
        "ensemble_total_nodes",
    ]
    pssr_results = [r for r in results if r["method"] == "PSSR"]
    ms_results = [r for r in results if r["method"] == "MULTI_SLIM"]
    datasets = sorted(set(r["dataset"] for r in results))
    
    def run_tests(scope_name: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        scope_stats: Dict[str, Any] = {}
        for metric in metrics:
            p_vals, m_vals = _paired_metric_values(pssr_results, ms_results, metric, dataset)
            if p_vals.size < 2:
                scope_stats[metric] = {
                    "ttest_stat": None,
                    "ttest_p": None,
                    "wilcoxon_stat": None,
                    "wilcoxon_p": None,
                    "mean_diff_ms_minus_pssr": None,
                    "n": int(p_vals.size),
                }
                continue
            t_stat, t_p = ttest_rel(p_vals, m_vals)
            try:
                w_stat, w_p = wilcoxon(p_vals, m_vals)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            diff_mean = float(np.mean(m_vals - p_vals))
            scope_stats[metric] = {
                "ttest_stat": float(t_stat),
                "ttest_p": float(t_p),
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
                "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
                "mean_diff_ms_minus_pssr": diff_mean,
                "n": int(p_vals.size),
            }
        return {"scope": scope_name, "metrics": scope_stats}
    
    overall = run_tests("overall", None)
    per_dataset = [run_tests(ds, ds) for ds in datasets]
    return {"overall": overall, "per_dataset": per_dataset}


def print_statistical_tests(stats: Dict[str, Any]) -> None:
    """Pretty-print statistical test results."""
    def format_scope(scope: Dict[str, Any]) -> None:
        print(f"\n[{scope['scope']}] Paired tests (MULTI_SLIM - PSSR):")
        for metric, vals in scope["metrics"].items():
            n = vals["n"]
            mean_diff = vals["mean_diff_ms_minus_pssr"]
            t_p = vals["ttest_p"]
            w_p = vals["wilcoxon_p"]
            print(f"  {metric}: n={n}, mean diff={mean_diff}")
            print(f"    t-test p={t_p}, wilcoxon p={w_p}")
    
    format_scope(stats["overall"])
    for scope in stats["per_dataset"]:
        format_scope(scope)


def run_comparison():
    results: List[Dict[str, Any]] = []
    total_runs = len(DATASETS) * SAMPLE_SIZE * 2  # 2 methods per sample
    current_run = 0
    cfg = CONFIG_LARGE
    
    print("=" * 80)
    print("PSSR vs MULTI_SLIM Comparison Test")
    print("=" * 80)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Configuration: {cfg['name']}")
    print(f"Samples per dataset: {SAMPLE_SIZE}")
    print(f"Total runs: {total_runs}")
    print("=" * 80)
    print()
    
    for dataset_name, loader in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        X, y = loader(X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        print()
        
        for sample_id in range(SAMPLE_SIZE):
            sample_seed = cfg["seed"] + sample_id
            run_cfg = {**cfg, "seed": sample_seed}
            print(f"  Sample {sample_id + 1}/{SAMPLE_SIZE} (seed={sample_seed})")
            print(f"  {'-'*76}")
            
            # PSSR
            current_run += 1
            print(f"    [{current_run}/{total_runs}] Running PSSR...", end=" ", flush=True)
            pssr_res = run_pssr(X_train, X_test, y_train, y_test, run_cfg)
            print(f"✓ ({pssr_res['time_sec']:.2f}s)")
            print(f"      Train - RMSE: {pssr_res['train_rmse']:.6f}, MAE: {pssr_res['train_mae']:.6f}, R²: {pssr_res['train_r2']:.6f}")
            print(f"      Test  - RMSE: {pssr_res['test_rmse']:.6f}, MAE: {pssr_res['test_mae']:.6f}, R²: {pssr_res['test_r2']:.6f}")
            print(f"      Ensemble nodes: {pssr_res['ensemble_nodes_count']}, total nodes: {pssr_res['ensemble_total_nodes']}")
            results.append(
                {
                    "dataset": dataset_name,
                    "config": cfg["name"],
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "method": "PSSR",
                    **pssr_res,
                }
            )
            
            # MULTI_SLIM
            current_run += 1
            print(f"    [{current_run}/{total_runs}] Running MULTI_SLIM...", end=" ", flush=True)
            ms_res = run_multi_slim(X_train, X_test, y_train, y_test, run_cfg)
            print(f"✓ ({ms_res['time_sec']:.2f}s)")
            print(f"      Train - RMSE: {ms_res['train_rmse']:.6f}, MAE: {ms_res['train_mae']:.6f}, R²: {ms_res['train_r2']:.6f}")
            print(f"      Test  - RMSE: {ms_res['test_rmse']:.6f}, MAE: {ms_res['test_mae']:.6f}, R²: {ms_res['test_r2']:.6f}")
            print(f"      Ensemble nodes: {ms_res['ensemble_nodes_count']}, total nodes: {ms_res['ensemble_total_nodes']}")
            
            # Comparison
            rmse_diff = ms_res['test_rmse'] - pssr_res['test_rmse']
            r2_diff = ms_res['test_r2'] - pssr_res['test_r2']
            print(f"      Comparison (MULTI_SLIM - PSSR):")
            print(f"        Test RMSE diff: {rmse_diff:+.6f} ({'PSSR better' if rmse_diff > 0 else 'MULTI_SLIM better' if rmse_diff < 0 else 'Equal'})")
            print(f"        Test R² diff:   {r2_diff:+.6f} ({'PSSR better' if r2_diff < 0 else 'MULTI_SLIM better' if r2_diff > 0 else 'Equal'})")
            print()
            
            results.append(
                {
                    "dataset": dataset_name,
                    "config": cfg["name"],
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "method": "MULTI_SLIM",
                    **ms_res,
                }
            )
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Calculate statistics by method
    pssr_results = [r for r in results if r['method'] == 'PSSR']
    ms_results = [r for r in results if r['method'] == 'MULTI_SLIM']
    
    def calc_stats(values):
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        mean = np.mean(values)
        std = np.std(values)
        return {'mean': mean, 'std': std, 'min': np.min(values), 'max': np.max(values)}
    
    print("\nAverage Test RMSE by Method:")
    pssr_rmse_stats = calc_stats([r['test_rmse'] for r in pssr_results])
    ms_rmse_stats = calc_stats([r['test_rmse'] for r in ms_results])
    print(f"  PSSR:       Mean={pssr_rmse_stats['mean']:.6f}, Std={pssr_rmse_stats['std']:.6f}, Min={pssr_rmse_stats['min']:.6f}, Max={pssr_rmse_stats['max']:.6f}")
    print(f"  MULTI_SLIM: Mean={ms_rmse_stats['mean']:.6f}, Std={ms_rmse_stats['std']:.6f}, Min={ms_rmse_stats['min']:.6f}, Max={ms_rmse_stats['max']:.6f}")
    
    print("\nAverage Test R² by Method:")
    pssr_r2_stats = calc_stats([r['test_r2'] for r in pssr_results])
    ms_r2_stats = calc_stats([r['test_r2'] for r in ms_results])
    print(f"  PSSR:       Mean={pssr_r2_stats['mean']:.6f}, Std={pssr_r2_stats['std']:.6f}, Min={pssr_r2_stats['min']:.6f}, Max={pssr_r2_stats['max']:.6f}")
    print(f"  MULTI_SLIM: Mean={ms_r2_stats['mean']:.6f}, Std={ms_r2_stats['std']:.6f}, Min={ms_r2_stats['min']:.6f}, Max={ms_r2_stats['max']:.6f}")
    
    print("\nAverage Execution Time by Method:")
    pssr_time_stats = calc_stats([r['time_sec'] for r in pssr_results])
    ms_time_stats = calc_stats([r['time_sec'] for r in ms_results])
    print(f"  PSSR:       Mean={pssr_time_stats['mean']:.2f}s, Std={pssr_time_stats['std']:.2f}s, Min={pssr_time_stats['min']:.2f}s, Max={pssr_time_stats['max']:.2f}s")
    print(f"  MULTI_SLIM: Mean={ms_time_stats['mean']:.2f}s, Std={ms_time_stats['std']:.2f}s, Min={ms_time_stats['min']:.2f}s, Max={ms_time_stats['max']:.2f}s")
    
    print("\nPer-Dataset Comparison (Test RMSE):")
    datasets = set(r['dataset'] for r in results)
    for dataset in sorted(datasets):
        dataset_pssr = [r for r in pssr_results if r['dataset'] == dataset]
        dataset_ms = [r for r in ms_results if r['dataset'] == dataset]
        pssr_rmse = np.mean([r['test_rmse'] for r in dataset_pssr])
        ms_rmse = np.mean([r['test_rmse'] for r in dataset_ms])
        print(f"  {dataset:20s} - PSSR: {pssr_rmse:.6f}, MULTI_SLIM: {ms_rmse:.6f}, Diff: {ms_rmse - pssr_rmse:+.6f}")
    
    # Statistical tests
    stats = compute_statistical_tests(results)
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    print_statistical_tests(stats)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    out_dir = Path(__file__).resolve().parent / "comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    stats_path = out_dir / "stats.json"

    import csv

    if results:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Results saved to:")
        print(f"    - {csv_path}")
        print(f"    - {json_path}")
        print(f"    - {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
