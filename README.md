# PSSR: Piecewise Specialist Symbolic Regression

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library implementing **Piecewise Specialist Symbolic Regression (PSSR)**, a two-phase evolutionary approach for symbolic regression that combines multiple specialist models through learned conditional routing.

## Overview

PSSR is based on the Multi-SLIM approach and extends traditional Genetic Programming (GP) by:

1. **Training Specialist Models**: Evolves a population of GP individuals, each specializing in different regions of the input space
2. **Evolving Ensemble Trees**: Creates conditional routing trees that intelligently combine specialists based on learned conditions

The resulting model partitions the input space and routes inputs to appropriate specialist models, often achieving better performance than single GP models.

## Features

- 🧬 **Two-Phase Evolution**: Separate evolution phases for specialists and ensemble routing
- 🎯 **Specialist Ensembles**: Multiple GP models that specialize in different data regions
- 🌳 **Conditional Routing**: Learned conditions that route inputs to appropriate specialists
- ⚡ **Performance Optimized**: Cached semantics for fast ensemble evaluation
- 🔄 **Warm Start Support**: Continue training from existing populations
- 📊 **scikit-learn Compatible**: Implements `RegressorMixin` for easy integration
- 🎲 **Reproducible**: Random state control for reproducible results
- 📈 **Comprehensive Logging**: Track fitness evolution and model statistics

## Installation

### Requirements

- Python >= 3.11
- NumPy >= 2.3.4
- scikit-learn >= 1.7.2
- pandas >= 2.3.3

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/PSSR.git
cd PSSR

# Install in development mode
pip install -e .

# Or install with poetry
poetry install
```

## Quick Start

### Basic Usage

```python
import numpy as np
from pssr import PSSRegressor

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(100)

# Create and fit the model
model = PSSRegressor(
    specialist_pop_size=50,
    ensemble_pop_size=50,
    random_state=42
)

model.fit(X, y, specialist_n_gen=50, ensemble_n_gen=50)

# Make predictions
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y):.4f}")
```

### With Train/Test Split

```python
# Split data
X_train, X_test = X[:70], X[70:]
y_train, y_test = y[:70], y[70:]

# Fit with test data for monitoring
model.fit(
    X_train, y_train,
    X_test=X_test, y_test=y_test,
    specialist_n_gen=50,
    ensemble_n_gen=50,
    verbose=1  # Show progress
)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
```

### Using GPRegressor (Base Model)

```python
from pssr import GPRegressor

# Create a standard GP regressor
gp_model = GPRegressor(
    population_size=100,
    max_depth=6,
    random_state=42
)

gp_model.fit(X, y, n_gen=100)
predictions = gp_model.predict(X)
```

## Architecture

### Two-Phase Approach

**Phase 1: Specialist Training**
- Evolves a population of GP individuals using standard GP operators
- Each individual becomes a potential specialist
- Uses the full population as specialists (configurable)

**Phase 2: Ensemble Evolution**
- Creates conditional trees that route inputs to specialists
- Condition trees are GP expressions that determine routing
- Ensemble trees combine specialist outputs based on conditions

### Key Components

- **`GPRegressor`**: Base Genetic Programming regressor
- **`PSSRegressor`**: Two-phase PSSR regressor
- **`Specialist`**: Wrapper for GP individuals with cached semantics
- **`Condition`**: GP tree used as a routing predicate
- **`EnsembleIndividual`**: Conditional tree structure combining specialists

## API Reference

### PSSRegressor

Main regressor class implementing the PSSR algorithm.

#### Parameters

**Specialist Parameters:**
- `specialist_pop_size` (int, default=100): Population size for specialist evolution
- `specialist_max_depth` (int, default=6): Maximum tree depth for specialists
- `specialist_init_depth` (int, default=2): Initial tree depth for specialists

**Ensemble Parameters:**
- `ensemble_pop_size` (int, default=100): Population size for ensemble evolution
- `ensemble_max_depth` (int, default=4): Maximum depth for ensemble trees
- `depth_condition` (int, default=3): Maximum depth for condition trees

**Variation:**
- `p_xo` (float, default=0.5): Probability of crossover vs mutation

**General:**
- `random_state` (int, default=42): Random seed for reproducibility
- `normalize` (bool, default=False): Whether to normalize input data
- `functions` (list[str] or FunctionSet, default=['add', 'sub', 'mul', 'div']): Function set for GP trees
- `condition_functions` (list[str] or FunctionSet, optional): Function set for conditions (defaults to `functions`)
- `constant_range` (float, default=1.0): Range for constant terminals
- `selector` (str or Callable, default='tournament'): Selection method

#### Methods

- `fit(X, y, X_test=None, y_test=None, specialist_n_gen=100, ensemble_n_gen=100, verbose=0, warm_start=False)`: Fit the model
- `predict(X)`: Make predictions
- `score(X, y)`: Calculate R² score (scikit-learn compatible)

#### Attributes

- `specialists_`: Dictionary of trained specialists
- `best_ensemble_`: Best ensemble individual
- `specialist_population_`: Final specialist population
- `ensemble_population_`: Final ensemble population
- `log_`: Training log with fitness history

### GPRegressor

Base Genetic Programming regressor.

#### Parameters

- `population_size` (int, default=100): Population size
- `max_depth` (int, default=6): Maximum tree depth
- `init_depth` (int, default=2): Initial tree depth
- `p_xo` (float, default=0.5): Crossover probability
- `random_state` (int, default=42): Random seed
- `normalize` (bool, default=False): Normalize data
- `functions` (list[str] or FunctionSet): Function set
- `constant_range` (float, default=1.0): Constant range
- `selector` (str or Callable, default='tournament'): Selection method

## Examples

### Custom Function Set

```python
from pssr.core.functions import FunctionSet

# Create custom function set
custom_functions = FunctionSet()
custom_functions.add_function("add", lambda x, y: x + y, 2)
custom_functions.add_function("mul", lambda x, y: x * y, 2)
custom_functions.add_function("sin", np.sin, 1)

model = PSSRegressor(functions=custom_functions)
```

### Warm Start

```python
# Initial training
model.fit(X, y, specialist_n_gen=50, ensemble_n_gen=50)

# Continue training
model.fit(X, y, specialist_n_gen=50, ensemble_n_gen=50, warm_start=True)
```

### Accessing Model Components

```python
# Get best specialist
best_specialist = model.specialist_population_.get_best_individual()

# Get ensemble tree structure
ensemble_tree = model.best_ensemble_.collection

# Access training log
fitness_history = model.log_['best_fitness']
```

## Performance Considerations

- **Cached Semantics**: Specialist outputs are pre-computed and cached for fast ensemble evaluation
- **Vectorized Operations**: Uses NumPy for efficient batch operations
- **Combined Train/Test**: Single array with slicing for efficient memory usage
- **Warm Start**: Continue training without reinitializing populations

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pssr.py

# Run with coverage
pytest --cov=src/pssr tests/
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PSSR in your research, please cite:

```bibtex
@software{pssr2024,
  title={PSSR: Piecewise Specialist Symbolic Regression},
  author={Amaral, Mateus},
  year={2024},
  url={https://github.com/yourusername/PSSR}
}
```

## Acknowledgments

- Based on the Multi-SLIM approach for piecewise symbolic regression
- Built on top of Genetic Programming principles
- Inspired by ensemble methods and specialist models

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainer.

---

**Note**: This library is under active development. API may change in future versions.
