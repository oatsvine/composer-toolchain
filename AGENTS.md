# Repository Guidelines

## Project Structure & Module Organization
- `src/composer_toolchain/` houses runtime code: `core.py` handles workspace orchestration, `score.py` exposes excerpt utilities, and `cli.py` provides the Typer interface surfaced via `python -m composer_toolchain.cli`.
- `tests/` mirrors the module layout (e.g., `test_core.py`, `test_score_parts.py`) and should be extended alongside new functionality.
- `tests/data/corpus/` stores sample Humdrum and MusicXML assets used in fixtures; avoid modifying originals—stage temporary derivatives under `/tmp` or a workspace copy.

## Build, Test, and Development Commands
- `python -m pip install -e .[dev]` sets up the editable package with pytest/pyright tooling.
- `pytest` runs the entire suite defined in `pyproject.toml` → `tool.pytest.ini_options`.
- `pyright` performs static type analysis on `composer_toolchain` and `tests`.
- `black composer_toolchain tests` enforces formatting; match the repo’s 88‑column default.
- `python -m composer_toolchain.cli --help` lists workspace commands such as `init-with-score` and `export-midi` for manual verification.

## Coding Style & Naming Conventions
- Follow Black formatting (4-space indents, trailing commas, double quotes where practical) and keep modules import-sorted.
- Favor typed function signatures and `pydantic` models for data contracts; prefer snake_case helpers and UpperCamelCase classes (see `ScoreSpec`, `PartInfo`).
- Keep CLI options descriptive and use Typer’s `Annotated` metadata for validation.
- Log with `loguru.logger` at info/debug levels instead of `print`.

## Testing Guidelines
- Add pytest modules alongside the feature under test (e.g., `src/composer_toolchain/core.py` → `tests/test_core.py`).
- Use descriptive test names like `test_merge_keeps_time_signatures` and rely on fixtures in `tests/conftest.py` for corpus paths.
- Run `pytest -k <focus>` during iteration, then full `pytest` + `pyright` before opening a PR; fail fast on coverage gaps around score mutations and workspace side effects.

## Commit & Pull Request Guidelines
- Commits follow short, imperative summaries (`Housekeeping in workspace directories`, `Round of cleanup`). Keep related changes squashed and note affected modules in the body if needed.
- PRs should link issues, describe workspace/test steps, and include CLI transcripts or MIDI diffs when behavior changes. Screenshot/score excerpts are encouraged for UI- or notation-facing adjustments.
- Ensure CI-critical commands (`pytest`, `pyright`, `black --check`) pass locally before requesting review.
