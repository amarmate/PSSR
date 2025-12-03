# Best Practices

## When Making Changes

1. Always fix import sorting issues when modifying files
2. Use absolute imports for local modules (e.g., `from core.misc import ...`) always
3. Keep linting rules relaxed - don't add unnecessary strictness
4. Maintain consistency with existing code style
5. Test code and benchmarks can use print statements freely

## When Adding New Code

- Follow existing patterns in the codebase
- Use absolute imports for local modules
- Keep functions focused and reasonably sized (complexity ≤ 15)
- Add type hints for public APIs

## When Refactoring

- Preserve existing functionality
- Update imports to use absolute paths
- Maintain backward compatibility where possible
- Update tests if API changes

## Common Patterns to Follow

1. **Import order**: stdlib → third-party → local (with blank lines)
2. **Print statements**: Acceptable for debugging/benchmarking
3. **Relaxed linting**: Don't enforce overly strict style rules
4. **Type hints**: Use when helpful, but not mandatory

## When testing code
Use the virtual environment for testing. Dont modify the syspath. Instead, do python -m pssr. ... to test the code while being inside the src folder.
