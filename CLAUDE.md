# CLAUDE.md

## What this is

Tools to calibrate a (scanning) transmission electron microscope, with a focus on 4D-STEM
overfocus imaging and recovering geometric calibration (rotation, handedness/flip, camera
length, scan pitch, descan error) from data. Scientific background: preprint
https://arxiv.org/abs/2403.08538. The worked example is `examples/stem_overfocus.ipynb`.

This is research software for scientific measurement data. Keep each processing/calibration
step explicit and documentable; never silently alter results that could be read as real
experimental output.

## Commands

```bash
# Install for development (extras: test, common, diffraction, gui)
pip install uv && uv pip install --system -e .[test,common,diffraction,gui]

# Run the full test suite with coverage
pytest --cov=microscope_calibration --cov-report=term tests/

# Run a single test file / test
pytest tests/test_model.py
pytest tests/test_model.py::test_name

# Numba-exercising coverage run. Most functional tests are very slow with JIT off,
# so the `with_numba` marker selects the few that must run interpreted for coverage.
NUMBA_DISABLE_JIT=1 pytest -m with_numba tests/

# Regenerate common/_generated.py after changing Model4DSTEM or GENERATED_FUNCTIONS,
# then commit it. tests/test_codegen.py fails if it is outdated.
python -m microscope_calibration.common.generate

# Lint / format (also wired into pre-commit)
ruff check src/ tests/
ruff format src/ tests/
pre-commit run --all-files

# Type check src/ (configured in pyproject.toml, runs in CI, not in pre-commit)
pyright
```

Requires Python >= 3.14. `prototypes/` is excluded from lint, tests, and packaging — it holds
exploratory notebooks and is not shipped code.

## Dependency extras

Dependencies are intentionally partitioned to keep the core (`common/model.py`) importable
without pulling in the whole scientific Python stack (packages: see `pyproject.toml`):

- **core** (always): JAX, SymPy, TemGymCore — the model itself
- **common**: LiberTEM, Numba, SciPy, optimizers — UDFs, simulator, and optimizers
- **diffraction**: CIF/crystallography
- **gui**: the interactive UI

Several deps are pinned to git `master`/branches (LiberTEM-panel-ui, diffsims)
pending upstream releases — see the `FIXME` comments in `pyproject.toml`. Flag any new
third-party dependency you introduce.

## Architecture

The whole library is organized around one central object, **`Model4DSTEM`**
(`common/model.py`), a `jax_dataclasses.pytree_dataclass` holding the full geometric
calibration of a 4D-STEM experiment (overfocus, scan pixel pitch, scan center/rotation,
camera length, detector pixel pitch/center/rotation, semi-convergence slope, flip factor, and
a 12-parameter `DescanError`). Because it is a JAX pytree, it is both differentiable and
traceable.

Key layers, bottom-up:

1. **`common/model.py`** — the model and the optical ray-trace. `Model4DSTEM.trace()` builds
   an ordered chain of `temgym_core` components (source → scanner → specimen → descanner →
   detector) and runs rays through them. The model code is written for SymPy;
   `Model4DSTEM.lambdify(modules)` derives a subclass with methods compiled for NumPy or JAX
   (autodiff/JIT). Coordinate conventions are strict: `PixelYX` (detector/scan pixel space,
   y-then-x) vs `CoordXY` (physical space, x-then-y). Helpers `scan_to_real`/`real_to_scan`/
   `detector_to_real`/`real_to_detector` convert between them. `adjust_*` / `derive()` return
   **new** models (immutable update style).

2. **`util/sympy.py`** — the machinery that makes (1) possible: a custom `lambdify` (built on
   `wrapt`) that recurses into nested dataclasses/pytrees (`PixelYX`, `Model4DSTEM`,
   `DescanError`) so a symbolic trace can be turned into a numeric function. Touch with care;
   the three-backend contract depends on it.

3. **`common/stem_overfocus.py`** — pixel-to-pixel transformation matrices (backward
   projection, detector correction) calculated from the generated `ray_transfer_matrix()`;
   raises `FittingError` when (near-)singular, e.g. in focus. Numba projection kernels.

4. **`udf/stem_overfocus.py`** — LiberTEM UDFs that apply (3) over a real dataset:
   `OverfocusUDF` (reconstruct an overfocus/real-space image) and `CorrectedPickUDF`. Models
   are passed wrapped in a `dict` so they can be mutated in place (LiberTEM issue #1780); the
   transformation matrices are calculated in `get_task_data()` and cached per model via
   `lru_cache`.

5. **`util/stem_overfocus_sim.py`** — forward simulator (e.g. `smiley()` test object,
   `project()`) that generates synthetic 4D-STEM data from a model, used heavily in tests and
   the example notebook.

6. **`util/optimize.py`** — inverse problem: fit calibration parameters to reference data
   (center-of-mass regressions, diffraction rings, matched points) with JAX autodiff and
   `optimistix` (`solve_*`). `normalize_descan_error` canonicalizes the (overparameterized)
   descan-error representation.

7. **`util/diffraction.py`** — computes diffraction two-theta angles from a CIF crystal
   structure (uses the `diffraction` extra); feeds known-geometry references into the
   optimizers.

8. **`ui.py`** — interactive Panel/Bokeh calibration GUI (`gui` extra).
   `CoordinateCorrectionLayout` wires sliders/tables to live re-projection so a user can tune
   calibration against a displayed dataset. `CalibratedDataset` (`dataset.py`, also importable
   from `ui`) pairs a dataset with a model.

### Mental model for making changes

- The central invariant is that `trace`/model code must stay valid for **SymPy symbols**, so
  that it can be derived and compiled symbolically. Avoid branching on values (`if x > 0`);
  use `_one`/`flip_factor` style tricks (constants threaded through) instead.
- Model functions are always SymPy-compatible and use SymPy math (`sym.cos`, `sym.pi`, ...);
  called directly on floats they may return SymPy numbers. For NumPy or JAX, use
  `Model4DSTEM.lambdify(modules)`, a cached subclass whose methods marked with
  `@lambdify_method` are compiled on first use (convert instances with `from_model()`), or
  wrap a dedicated function with `@lambdify(modules=np|jnp, recurse_for=...)`, e.g. for
  gradients (see `examples/model.ipynb`). Create lambdified functions once, e.g. at module
  level: each new function object triggers a new symbolic derivation, so don't decorate inside
  loops or frequently called functions. Code that evaluates model methods with SymPy symbols,
  e.g. derivations, must use the symbolic class (`_symbolic_class()`).
- `trace()` is the single source of truth. Derive relations from it symbolically instead of
  hard-coding them: register the derivation in a `GENERATED_FUNCTIONS` registry (collected in
  `common/generate.py`) and run the generator. Never edit `common/_generated.py` by hand.
- Generated code is only valid for `Model4DSTEM` itself; new derived functions also need the
  runtime path for subclasses. See the `Model4DSTEM` docstring.
- Respect the `PixelYX` vs `CoordXY` distinction; mixing axis order is the most likely source
  of silent geometric bugs.
- `equinox.filter_jit`, which optimistix uses, treats Python numbers as static: each new value
  compiles again. Models and `PixelYX` hold Python floats, so convert arguments with
  `jax.tree.map(jnp.asarray, args)` before passing them, as `_least_squares()` in
  `util/optimize.py` does. `jax.jit` traces Python numbers and is not affected.
- pyright findings that come from the types of dependencies or from the `float` annotations
  of the model are silenced per line with `# pyright: ignore[rule]  # reason`, or per file
  for the GUI and `util/diffraction.py`. Unnecessary ignores are errors.
- JAX x64 is enabled at import time (`jax.config.update("jax_enable_x64", True)`) in every
  module that uses JAX — this must happen before other JAX imports, hence the
  `# ruff: disable[E402]` markers at the top of those files. JAX-based solvers call
  `check_float64()` (`util/optimize.py`).

## Testing notes

- `tests/conftest.py` provides a `random_model` fixture (randomized `Model4DSTEM`) used for
  property/round-trip style tests.
- The `with_numba` marker (declared in `pytest.ini`) tags tests that exercise Numba kernels so
  they can be re-run with `NUMBA_DISABLE_JIT=1` for coverage of the interpreted paths.
- CI (`.github/workflows/`) runs on Python 3.14 across Linux/Windows/macOS with all extras
  installed.

## Commit messages and PR descriptions

Keep them short and concise, for human review:

- Open with one sentence: what is replaced by what, and the reason if it isn't obvious.
- One-line bullets naming the change, not how it works. No internal function names,
  thresholds, defaults, timing or size numbers, or rationale already in docstrings.
- Point to docs ("see docstring", "see README") instead of explaining.
- Keep: attribution with a link to the source, user-visible behavior changes (one line each),
  minimal review notes.
- Leave out what reviewers see in the diff or trust CI for, e.g. "existing tests unchanged".
- Don't instruct reviewers ("don't review X", "start with Y"); give useful context instead,
  e.g. "`X` is generated code, `test_abc()` checks that it is up to date".
- Commit messages: a short subject plus a few lines at most. Use the
  "Changes / Behavior changes / Review notes" structure only for PRs.
