"""
Preset configurations for PSSRegressor.

Presets provide pre-configured parameter sets for common PSSR use cases,
configuring both the specialist and ensemble phases.
"""


def fetch_pss_preset(preset_name: str) -> dict:
    """
    Fetch preset parameters for PSSR components.

    Parameters
    ----------
    preset_name : str
        Name of the preset to fetch.

    Returns
    -------
    dict
        Preset parameters with the following possible keys:

        Shared parameters:
            - "functions" : List of function names for both phases
            - "constant_range" : Range for constant generation
            - "p_xo" : Crossover probability (applies to both phases)

        Specialist phase parameters (prefixed with specialist_):
            - "specialist_pop_size" : Population size for specialists
            - "specialist_max_depth" : Maximum tree depth for specialists
            - "specialist_init_depth" : Initial depth for specialist trees
            - "specialist_selector" : Selection method for specialists
            - "specialist_initializer" : Initialization method for specialists
            - "specialist_crossover" : Crossover method for specialists
            - "specialist_mutation" : Mutation method for specialists
            - "specialist_selector_args" : Additional selector arguments
            - "specialist_initializer_args" : Additional initializer arguments
            - "specialist_crossover_args" : Additional crossover arguments
            - "specialist_mutation_args" : Additional mutation arguments

        Ensemble phase parameters (prefixed with ensemble_):
            - "ensemble_pop_size" : Population size for ensemble
            - "ensemble_max_depth" : Maximum depth for ensemble trees
            - "depth_condition" : Maximum depth for condition trees
            - "ensemble_selector" : Selection method for ensemble
            - "ensemble_crossover" : Crossover method for ensemble
            - "ensemble_selector_args" : Additional selector arguments

        Condition functions (for ensemble routing):
            - "condition_functions" : Functions for condition trees (defaults to functions)

    Raises
    ------
    ValueError
        If preset_name is not found in available presets.

    Examples
    --------
    >>> preset = fetch_pss_preset("default")
    >>> print(preset["specialist_pop_size"])
    100
    """
    presets = {
        "default": {
            # Shared
            "functions": ["add", "sub", "mul", "div"],
            "constant_range": 1.0,
            "p_xo": 0.5,
            # Specialist phase
            "specialist_pop_size": 100,
            "specialist_max_depth": 6,
            "specialist_init_depth": 2,
            "specialist_selector": "tournament",
            "specialist_initializer": "rhh",
            "specialist_crossover": "single_point",
            "specialist_mutation": "subtree",
            "specialist_selector_args": {"pool_size": 3},
            # Ensemble phase
            "ensemble_pop_size": 100,
            "ensemble_max_depth": 4,
            "depth_condition": 3,
            "ensemble_selector": "tournament",
            "ensemble_crossover": "homologous",
            "ensemble_selector_args": {"pool_size": 3},
        },
        "small": {
            # Shared
            "functions": ["add", "sub", "mul", "div"],
            "constant_range": 1.0,
            "p_xo": 0.5,
            # Specialist phase - smaller for quick experiments
            "specialist_pop_size": 50,
            "specialist_max_depth": 4,
            "specialist_init_depth": 2,
            "specialist_selector": "tournament",
            "specialist_initializer": "rhh",
            "specialist_crossover": "single_point",
            "specialist_mutation": "subtree",
            "specialist_selector_args": {"pool_size": 2},
            # Ensemble phase
            "ensemble_pop_size": 50,
            "ensemble_max_depth": 3,
            "depth_condition": 2,
            "ensemble_selector": "tournament",
            "ensemble_crossover": "homologous",
            "ensemble_selector_args": {"pool_size": 2},
        },
        "large": {
            # Shared
            "functions": ["add", "sub", "mul", "div"],
            "constant_range": 1.0,
            "p_xo": 0.6,
            # Specialist phase - larger for complex problems
            "specialist_pop_size": 200,
            "specialist_max_depth": 8,
            "specialist_init_depth": 3,
            "specialist_selector": "tournament",
            "specialist_initializer": "rhh",
            "specialist_crossover": "single_point",
            "specialist_mutation": "subtree",
            "specialist_selector_args": {"pool_size": 5},
            # Ensemble phase
            "ensemble_pop_size": 200,
            "ensemble_max_depth": 5,
            "depth_condition": 4,
            "ensemble_selector": "tournament",
            "ensemble_crossover": "homologous",
            "ensemble_selector_args": {"pool_size": 5},
        },
        "lexicase": {
            # Shared
            "functions": ["add", "sub", "mul", "div"],
            "constant_range": 1.0,
            "p_xo": 0.5,
            # Specialist phase with lexicase selection
            "specialist_pop_size": 100,
            "specialist_max_depth": 6,
            "specialist_init_depth": 2,
            "specialist_selector": "eplex",
            "specialist_initializer": "rhh",
            "specialist_crossover": "single_point",
            "specialist_mutation": "subtree",
            "specialist_selector_args": {},
            # Ensemble phase with lexicase
            "ensemble_pop_size": 100,
            "ensemble_max_depth": 4,
            "depth_condition": 3,
            "ensemble_selector": "eplex",
            "ensemble_crossover": "homologous",
            "ensemble_selector_args": {},
        },
    }

    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"Preset '{preset_name}' not found. Available presets: {available}"
        )

    return presets[preset_name]


def list_pss_presets() -> list[str]:
    """
    List all available PSSR preset names.

    Returns
    -------
    list[str]
        List of available preset names.
    """
    return ["default", "small", "large", "lexicase"]

