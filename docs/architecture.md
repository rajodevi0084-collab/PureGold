# Architecture Overview

This project targets Apple Silicon developer laptops using Python 3.11 from python.org, the `uv` package manager, and native ARM64 wheels wherever possible.

## Runtime Baseline
- **Python distribution**: Install the official Python 3.11.x universal2 installer from python.org to ensure consistent framework builds and access to the latest security patches.
- **Package manager**: Use `uv` for dependency resolution, locking, and execution. It offers deterministic builds and a familiar workflow for developers coming from `pip` or `pipenv`.
- **Native wheels**: Prefer ARM64 wheels for all dependencies so that M-series laptops avoid Rosetta translation penalties and keep parity with CI runners.

## Tooling
- `uv sync` provisions the virtual environment from `pyproject.toml` / `uv.lock`.
- `uv run` executes commands within the managed environment (e.g., linting, smoke tests).

## Continuous Integration
- CI uses macOS 13 and macOS 14 runners to exercise the same Python 3.11 universal2 build and `uv` workflow, ensuring that deterministic dependency resolution holds for the team and in automation.
