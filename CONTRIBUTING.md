# Contributing

## Getting Started

1. Clone the repo and install dependencies:

   ```bash
   uv sync
   ```

2. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

   This sets up automatic formatting (`ruff format`) and linting (`ruff check`) on every commit.

## Code Quality

- Format with `ruff format` and lint with `ruff check --fix`
- Use NumPy-style docstrings

## Git

- Use conventional commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`)
