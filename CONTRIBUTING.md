# Contributing to PSSR

Thank you for your interest in contributing to PSSR! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/PSSR.git
   cd PSSR
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/originalowner/PSSR.git
   ```

## Development Setup

### Prerequisites

- Python >= 3.11
- Poetry (recommended) or pip
- Git

### Installation

```bash
# Install dependencies with poetry
poetry install

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Development Dependencies

The project uses:
- **pytest**: Testing framework
- **ruff**: Fast Python linter
- **black**: Code formatter
- **mypy**: Type checking (optional)

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure

### Before You Start

1. **Check existing issues**: Look for open issues that match your contribution
2. **Create an issue**: For major changes, create an issue first to discuss
3. **Check the roadmap**: See if your contribution aligns with project goals

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use NumPy-style docstrings

### Formatting

We use **Black** for code formatting and **ruff** for linting:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Optional
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def example_function(
    X: Array,
    y: Optional[Array] = None,
) -> float:
    """Example function with type hints."""
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of the function.
    
    Longer description explaining what the function does,
    its purpose, and any important details.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
        
    Returns
    -------
    return_type
        Description of return value
        
    Raises
    ------
    ValueError
        When something goes wrong
        
    Examples
    --------
    >>> example_usage()
    result
    """
    ...
```

## Testing

### Writing Tests

- Write tests for all new features
- Aim for high test coverage
- Use descriptive test names
- Follow the existing test structure

### Test Structure

```python
def test_feature_name():
    """
    Test description explaining what is being tested.
    """
    # Arrange
    ...
    
    # Act
    ...
    
    # Assert
    ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pssr.py

# Run with coverage
pytest --cov=src/pssr --cov-report=html

# Run specific test
pytest tests/test_pssr.py::test_specific_function

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s
```

### Test Requirements

- All tests must pass before submitting a PR
- New code should include tests
- Maintain or improve test coverage
- Tests should be fast and independent

## Documentation

### Code Documentation

- Document all public functions and classes
- Include parameter descriptions
- Provide usage examples in docstrings
- Update docstrings when changing code

### README Updates

- Update README.md for user-facing changes
- Add examples for new features
- Update installation instructions if needed

### API Documentation

- Keep API documentation up to date
- Document breaking changes
- Provide migration guides for major changes

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make your changes**:
   - Write code following the style guide
   - Add tests for new features
   - Update documentation
   - Ensure all tests pass

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

   Use clear, descriptive commit messages:
   - Start with a verb (Add, Fix, Update, Remove, etc.)
   - Be specific about what changed
   - Reference issue numbers if applicable

   Examples:
   - `Add support for custom fitness functions`
   - `Fix bug in ensemble semantics calculation`
   - `Update documentation for PSSRegressor`

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Checklist

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] PR description explains the changes

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
Describe how you tested your changes.

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
```

## Project Structure

```
PSSR/
├── src/
│   └── pssr/
│       ├── __init__.py          # Main package exports
│       ├── core/                 # Core functionality
│       │   ├── primitives.py    # Primitive sets
│       │   ├── functions.py      # GP functions
│       │   ├── selection.py      # Selection operators
│       │   └── representations/ # Individual and Population
│       ├── gp/                   # Genetic Programming
│       │   ├── gp_regressor.py  # Base GP regressor
│       │   ├── gp_evolution.py  # Evolution loop
│       │   └── ...
│       └── pssr_model/           # PSSR specific
│           ├── pss_regressor.py # Main PSSR class
│           ├── specialist.py    # Specialist wrapper
│           ├── condition.py      # Condition trees
│           └── ...
├── tests/                        # Test suite
├── docs/                         # Documentation (if applicable)
├── README.md                     # Project readme
├── CONTRIBUTING.md               # This file
├── LICENSE                       # License file
└── pyproject.toml               # Project configuration
```

### Key Components

- **`core/`**: Core GP infrastructure (primitives, selection, representations)
- **`gp/`**: Standard Genetic Programming implementation
- **`pssr_model/`**: PSSR-specific components (specialists, ensembles, conditions)
- **`tests/`**: Test files mirroring the source structure

## Development Workflow

### Adding a New Feature

1. Create an issue to discuss the feature
2. Create a feature branch
3. Implement the feature with tests
4. Update documentation
5. Submit a PR

### Fixing a Bug

1. Create an issue describing the bug
2. Create a fix branch
3. Write a test that reproduces the bug
4. Fix the bug
5. Ensure the test passes
6. Submit a PR

### Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Questions?

If you have questions about contributing:

- Open an issue with the `question` label
- Check existing issues and discussions
- Review the codebase and documentation

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md (if applicable)
- Release notes
- Project documentation

Thank you for contributing to PSSR! 🎉

