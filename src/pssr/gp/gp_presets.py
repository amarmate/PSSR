

def fetch_preset(preset_name: str) -> dict:
    """Fetch preset parameters for GP components.

    Args:
        preset_name (str): Name of the preset.
    Returns:
        dict: Preset parameters with the following possible keys:
            - "functions" : List of function names to use in the primitive set.
            - "constant_range" : Range for constant generation.
            - "population_size" : Population size for the GP.
            - "n_gen" : Number of generations to run.
            - "max_depth" : Maximum depth of the GP trees.
            - "init_depth" : Initial depth for tree generation.
            - "p_xo" : Crossover probability.
            - "selector" : Selection method name.
            - "initializer" : Initialization method name.
            - "crossover" : Crossover method name.
            - "mutation" : Mutation method name.
            - "selector_args" : Additional arguments for the selector.
            - "initializer_args" : Additional arguments for the initializer.
            - "crossover_args" : Additional arguments for the crossover.
            - "mutation_args" : Additional arguments for the mutation.
    """
    
    presets = {}
    
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(presets.keys())}")

    return presets[preset_name]
    