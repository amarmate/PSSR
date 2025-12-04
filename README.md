# PSSR: Piecewise Specialist Symbolic Regression

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library implementing **Piecewise Specialist Symbolic Regression (PSSR)**, a two-phase evolutionary approach for symbolic regression that combines multiple specialist models through learned conditional routing.

## Overview

PSSR is a piecewise function approximation method that extends traditional Genetic Programming (GP) by:

1. **Training Specialist Models**: Evolves a population of GP individuals using lexicase selection, each specializing in different regions of the input space
2. **Evolving Ensemble Trees**: Creates conditional routing trees that intelligently combine specialists based on learned conditions

The resulting model partitions the input space and routes inputs to appropriate specialist models, often achieving better performance than single GP models.

For more details and the original implementation, see [my thesis](http://hdl.handle.net/10362/190604).

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

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the project.

### Citation

If you use PSSR in your research, please cite:

```
@mastersthesis{Amaral2025PSSR,
  title        = {Piecewise Symbolic Regression via Lexciase-Guided Specialist Ensembles},
  author       = {Amaral, Mateus Baptista},
  advisor      = {Vanneschi, Leonardo},
  school       = {NOVA Information Management School (NOVA IMS)},
  year         = {2025},
  type         = {Master's Thesis},
  url          = {http://hdl.handle.net/10362/190604}
}
```
