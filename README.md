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

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the project.

## Citation

If you use PSSR in your research, please cite:

```bibtex
@mastersthesis{Amaral2025PSSR,
  title        = {Piecewise Symbolic Regression via Lexciase-Guided Specialist Ensembles},
  author       = {Amaral, Mateus Baptista},
  advisor      = {Vanneschi, Leonardo},
  school       = {NOVA Information Management School (NOVA IMS)},
  year         = {2025},
  type         = {Master's Thesis},
  url          = {http://hdl.handle.net/10362/190604}
}
