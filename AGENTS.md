# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/research/` with subpackages for `llms/` (OpenAI, Gemini, HuggingFace, utils), `torch/` (MNIST), `stats/` (logistic regression), `modal/`, and `papers/` (experiments).
- Data: `data/` for datasets and artifacts (e.g., MNIST downloads). Keep large files out of Git.
- Config: `pyproject.toml` (Python >=3.12, deps). Local venv: `.venv/` (optional). Runtime logs may write under `src/research/logs`.

## Build, Test, and Development Commands
- Environment (recommended, uv): `uv sync` then `uv run python -V` to verify.
- Install (pip alternative): `python -m venv .venv && source .venv/bin/activate && pip install -e .`.
- Run examples:
  - MNIST: `PYTHONPATH=src python -m research.torch.mnist`
  - Gemini quickstart (requires `GCP_PROJECT_ID`): `PYTHONPATH=src python src/research/llms/gemini/quickstart.py`
  - Modal example: `modal run src/research/modal/hello_world.py` (requires Modal auth).

## Coding Style & Naming Conventions
- Python: 4‑space indent, type hints where practical, concise docstrings for public functions.
- Naming: modules/files `snake_case`, functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Lint/format: Prefer `ruff` if available: `uvx ruff check .` and `uvx ruff format .`. Keep diffs minimal and focused.

## Testing Guidelines
- Framework: `pytest` (add to dev deps if missing).
- Layout: `tests/` mirroring `src/research/...` paths.
- Naming: files `test_*.py`; tests `test_*` with clear arrange/act/assert.
- Run: `uvx pytest -q` or `pytest -q` inside the venv. No strict coverage requirement yet; include regression tests for new bugs.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (e.g., "add torch example"); add a body when changing behavior or configs.
- PRs: include summary, rationale, run/repro steps, sample output, affected modules, and any env vars/configs touched. Link issues and add screenshots for notebooks/results when helpful.

## Security & Configuration Tips
- Secrets via `.env` (dotenv is used). Examples: `GCP_PROJECT_ID`, Langfuse keys for Gemini + Instructor, API keys for LLM providers. Never commit secrets.
- Data lives in `data/`; keep large datasets and generated artifacts out of Git (see `.gitignore`).
- For Modal: authenticate with `modal token set` before `modal run`.
