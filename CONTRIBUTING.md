# Contributing

## Getting Started

1. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/davenpi/computer.git
   cd computer
   uv sync
   ```

2. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

   This sets up automatic formatting (`ruff format`) and linting (`ruff check`) on every commit.

## Testing

- Run tests with `pytest` (pre-commit runs them automatically on commit)
- Add tests for new functionality in `tests/`

## Code Quality

- Format with `ruff format` and lint with `ruff check --fix`
- Use NumPy-style docstrings

## Git

- Use conventional commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`)
