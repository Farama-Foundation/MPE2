# MPE2 contributor agent instructions

## Repository structure in `mpe2/`

- `docs/` contains documentation source and scripts for generating environment docs.
- `mpe2/_mpe_utils/` contains shared simulation primitives and environment base helpers.
- `mpe2/<env_name>/<env_name>.py` contains the actual implementation for an environment family.
- `mpe2/<env_id>.py` is the public versioned shim that re-exports `env`, `parallel_env`, and `raw_env` from the implementation folder.
- `mpe2/all_modules.py` is the environment registry used for imports by id.
- `mpe2/__init__.py` only defines package metadata (for example `__version__`).
- `test/` contains tests code.

## How to add a new environment

- Use a base environment directory name without version suffix (for example `simple_example`) and a module id with suffix (for example `simple_example_v1`).
- Put implementation logic in `mpe2/simple_example/simple_example.py`.
- Keep the top of this file as a markdown-style module docstring: this is the source of environment docs.
- Add a version shim `mpe2/simple_example_v1.py`:

```python
from mpe2.simple_example.simple_example import env, parallel_env, raw_env

__all__ = ["env", "parallel_env", "raw_env"]
```

- Register the env in `mpe2/all_modules.py` by adding the import and a `mpe/<id>` entry to `mpe_environments`.
- For additional variants, add additional shim files (for example `simple_example_v2.py`) and registry entries.

## How to update docs when adding or updating envs

- Add an icon in `docs/_static/img/icons/` named exactly `<env_name>.png`.
- Add a preview gif in `docs/environments/mpe2/` named `mpe2_<env_name>.gif`.
- Regenerate the API/doc page from docstrings:

```bash
python docs/_scripts/gen_envs_mds.py
```

- Regenerate the environment grid/list page:

```bash
python docs/_scripts/gen_envs_display.py
```

- Edit `docs/index.md` and `docs/mpe2.md` to include the new env in navigation (`toctree`).
- Edit `docs/_scripts/gen_envs_display.py` `all_envs["mpe2"]` if you keep that list in sync manually.
- Regenerate/commit the generated page output in `docs/environments/mpe2/list.html` after script changes.

## Naming and path rules to avoid mismatches

- The registry id in `all_modules.py` uses the versioned name (for example `mpe/simple_example_v1`).
- The generated docs source file should be under `docs/environments/` and use the base name (for example `docs/environments/simple_example.md`).
- The gif used in docs pages follows `mpe2/environments/mpe2_<base_name>.gif` in generated links.

## Running tests
Run tests after every change to ensure nothing is broken:
```sh
pytest -s tests
```

## Running pre-commit hooks
Pre-commit hooks are used to enforce code formatting and linting for Python code (Black, isort, flake8, pyright, codespell).

To install pre-commit hooks, run:
```sh
pre-commit install
```

To run pre-commit hooks manually, use:
```sh
pre-commit run --all-files
```
