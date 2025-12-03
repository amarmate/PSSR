# Code Style & Formatting

## Code Formatting

- Use Black for formatting (line length: 88)
- Follow Ruff linting rules as configured in `pyproject.toml`
- The linting rules are intentionally relaxed - focus on critical errors, not style preferences

## Naming Conventions

- Follow PEP 8 naming conventions
- Function/argument names should be lowercase
- Class names should use CapWords
- Variable names can be single letters when contextually clear (e.g., `s`, `m`, `n`)

## Output & Debugging

- Use `print()` statements for output and debugging (not logging)
- Print statements are acceptable in test/benchmark code and `if __name__ == "__main__"` blocks

## Type Hints

- Use type hints where appropriate
- Use `numpy.typing` for array types: `npt.NDArray[np.float64]`
- Type hints are encouraged but not strictly enforced

## What NOT to Do

- Don't add overly strict linting rules
- Don't use relative imports 
- The test folder should never modify sys.path. Sys.path should never be modified!
- Don't enforce naming conventions that conflict with mathematical notation (e.g., single-letter variables are okay)
- Don't add unnecessary complexity to satisfy linters

