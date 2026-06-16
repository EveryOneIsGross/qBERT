# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python scripts for qBERT/qGPT experimentation, chat inference, dataset auditing, and model training. Primary entry points live at the repository root: `qBERT.py`, `qGPT2.py`, `CHATbert.py`, `CHATgpt.py`, `autoCHATbert.py`, `trainer.py`, `trainer_st.py`, `validate_trainer.py`, and `audit_dataset.py`. Configuration files are in `config/`; visual/static assets are in `assets/`. Model outputs, tokenizer files, checkpoints, TensorBoard runs, and generated artifacts are stored under directories such as `qbert-hermes*/`, `qbert-st-hermes-v4/`, and `checkpoints/`. Treat `archive/`, `cache/`, `logs/`, and `__pycache__/` as generated or historical material.

## Build, Test, and Development Commands

Use Python commands from the repository root so relative config and model paths resolve correctly.

- `python qBERT.py`: run the main qBERT script.
- `python CHATbert.py`: start the BERT chat workflow.
- `python trainer.py`: run the main training pipeline.
- `python trainer_st.py`: run the sentence-transformer training variant.
- `python validate_trainer.py`: validate trainer behavior and configuration.
- `python audit_dataset.py`: inspect dataset quality before training.

There is no committed package manifest. Document new dependencies in `readme.md` and prefer a virtual environment for local installs.

## Coding Style & Naming Conventions

Follow standard Python style with 4-space indentation, descriptive snake_case names for functions and variables, and PascalCase only for classes. Guard runnable code with `if __name__ == "__main__":` when adding new modules. Prefer `config/*.yaml` over hard-coded paths or model parameters. Avoid adding files with spaces in their names; existing `copy` files are snapshots, not naming examples.

## Testing Guidelines

Use `validate_trainer.py` for trainer checks and add focused tests near changed behavior. Name new test files `test_*.py` if introducing a test framework. For training changes, run dataset auditing first, then a small validation or short training pass before long GPU jobs. Do not commit bulky checkpoints, logs, or cache files unless they are release artifacts.

## Commit & Pull Request Guidelines

Keep commits focused and use short imperative subjects, for example `Add trainer config validation` or `Update chat prompt loading`. Pull requests should describe the change, list commands run, mention affected model/config paths, and include screenshots only when UI or asset output changes. Call out required local data, checkpoint, or GPU assumptions.

## Security & Configuration Tips

Keep credentials, API keys, local datasets, and private model weights out of source control. Store machine-specific paths in local config overrides or environment variables. Before committing, check `git status --short` for accidental checkpoint, cache, or log additions.
