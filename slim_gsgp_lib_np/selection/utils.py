import random
import math
import json
import os

CACHE_FILE = os.path.join('slim_gsgp_lib_np', 'selection', 'musigma_cache.json')

def get_musigma_from_cache(pp: float):
    int_pp = int(pp) 
    if pp > 78: 
        pp = 78 
    key = f"{int(pp)}.0"
    
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache-Datei nicht gefunden: {CACHE_FILE}")

    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)

    if key not in cache:
        raise KeyError(f"pp-Wert {key} nicht im Cache (gültige Keys: {list(cache.keys())[:5]} …).")

    mean_count, std_count = cache[key]
    return mean_count, std_count








# def calculate_musigma_pp_cached(pp: float, n_cases=500, n_samples=500):
#     if os.path.exists(CACHE_FILE):
#         with open(CACHE_FILE, "r") as f:
#             cache = json.load(f)
#             # Convert string keys to float
#             cache = {float(k): tuple(v) for k, v in cache.items()}
#     else:
#         cache = {}

#     # If cache is empty or pp not present, compute and store
#     if pp not in cache:
#         counts = []
#         for _ in range(n_samples):
#             I = [random.gauss(0, pp) for _ in range(n_cases)]
#             max_I = max(I)
#             exp_I = [math.exp(i - max_I) for i in I]
#             total = sum(exp_I)
#             weights = [e / total for e in exp_I]
#             count = sum(1 for w in weights if w >= 0.01)
#             counts.append(count)
#         mean_count = sum(counts) / n_samples
#         var = sum((c - mean_count) ** 2 for c in counts) / (n_samples - 1)
#         std_count = math.sqrt(var)
#         cache[pp] = (mean_count, std_count)

#         # Save updated cache
#         with open(CACHE_FILE, "w") as f:
#             json.dump({str(k): v for k, v in cache.items()}, f)

#     # Return closest value in cache
#     closest_pp = min(cache.keys(), key=lambda x: abs(x - pp))
#     return cache[closest_pp]
