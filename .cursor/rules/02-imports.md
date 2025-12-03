# Import Rules

## Import Organization

- Do not use relative imports for local modules (e.g., use `from core.misc import ...` instead of `from .misc import ...`)
- Follow Ruff's isort rules: standard library → third-party → local imports
- Group imports with blank lines between groups
- Alphabetize imports within each group

## Import Order

1. **Standard library imports** (alphabetized)
2. **Blank line**
3. **Third-party imports** (alphabetized)
4. **Blank line**
5. **Local imports** (alphabetized)

## When Making Changes

- Always fix import sorting issues when modifying files
- Use absolute imports for local modules (e.g., `from core.misc import ...`)
- Update imports to use absolute paths when refactoring

