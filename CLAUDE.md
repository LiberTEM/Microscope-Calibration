# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Lint / format (also wired into pre-commit)
ruff check src/ tests/
ruff format src/ tests/
pre-commit run --all-files
```

Requires Python >= 3.14. `prototypes/` is excluded from lint, tests, and packaging — it holds
exploratory notebooks and is not shipped code.

## Dependency extras

Dependencies are intentionally partitioned to keep the core (`common/model.py`) importable
without pulling in the whole scientific Python stack:

- **core** (always): `jax`, `jax-dataclasses`, `sympy`, `temgym-core`, `flax`, `wrapt`, `frozendict`
- **common**: `libertem`, `numba`, `scipy`, `scikit-image`, `optax`, `optimistix`, `numpy` — needed by the UDFs, simulator, and optimizers
- **diffraction**: CIF/crystallography (`pycifrw`, `diffpy.structure`, `orix`, `diffsims`)
- **gui**: `libertem-ui`, `panel`, `bokeh`, `ipywidgets`, etc. — the interactive UI

Several deps are pinned to git `master`/branches (TemGymCore, LiberTEM-panel-ui, diffsims)
pending upstream releases — see the `FIXME` comments in `pyproject.toml`. Flag any new
third-party dependency you introduce.

## Architecture

The whole library is organized around one central object, **`Model4DSTEM`**
(`common/model.py`), a `jax_dataclasses.pytree_dataclass` holding the full geometric
calibration of a 4D-STEM experiment (overfocus, scan pixel pitch, scan center/rotation,
camera length, detector pixel pitch/center/rotation, semi-convergence angle, flip factor, and
a 12-parameter `DescanError`). Because it is a JAX pytree, it is both differentiable and
traceable.

Key layers, bottom-up:

1. **`common/model.py`** — the model and the optical ray-trace. `Model4DSTEM.trace()` builds
   an ordered chain of `temgym_core` components (source → scanner → specimen → descanner →
   detector) and runs rays through them. The same code path runs on three backends:
   concrete `float` values, `sympy` symbols (symbolic model), and `jax` tracers
   (autodiff/JIT). Coordinate conventions are strict: `PixelYX` (detector/scan pixel space,
   y-then-x) vs `CoordXY` (physical space, x-then-y). Helpers `scan_to_real`/`real_to_scan`/
   `detector_to_real`/`real_to_detector` convert between them. `adjust_*` / `derive()` return
   **new** models (immutable update style). `lambdify_trace_for(modules)` compiles the
   symbolic trace to a fast callable for a given numeric backend.

2. **`util/sympy.py`** — the machinery that makes (1) possible: a custom `lambdify` (built on
   `wrapt`) that recurses into nested dataclasses/pytrees (`PixelYX`, `Model4DSTEM`,
   `DescanError`) so a symbolic trace can be turned into a numeric function. Touch with care;
   the three-backend contract depends on it.

3. **`common/stem_overfocus.py`** — derives the pixel-to-pixel transformation matrices
   (forward and backward projection, detector correction) from a model by least-squares
   fitting against the trace. Asserts the model is actually linear (`_do_lstsq`); raises
   `FittingError` when ill-posed. Contains the Numba-accelerated projection kernels.

4. **`udf/stem_overfocus.py`** — LiberTEM UDFs that apply (3) over a real dataset:
   `OverfocusUDF` (reconstruct an overfocus/real-space image) and `CorrectedPickUDF`. Models
   are passed wrapped in a `dict` so they can be mutated in place (LiberTEM issue #1780); the
   transformation matrices are cached via `lru_cache` and only recomputed when the model
   changes.

5. **`util/stem_overfocus_sim.py`** — forward simulator (e.g. `smiley()` test object,
   `project()`) that generates synthetic 4D-STEM data from a model, used heavily in tests and
   the example notebook.

6. **`util/optimize.py`** — inverse problem. Given reference data (center-of-mass regressions,
   diffraction rings, matched points), fit individual calibration parameters back out:
   `solve_camera_length`, `solve_scan_pixel_pitch`, `solve_full_descan_error`,
   `solve_tilt_descan_error`, `solve_coords_points`, etc. Built on JAX autodiff with
   `optax`/`optimistix`. `normalize_descan_error` canonicalizes the (overparameterized)
   descan-error representation.

7. **`util/diffraction.py`** — computes diffraction two-theta angles from a CIF crystal
   structure (uses the `diffraction` extra); feeds known-geometry references into the
   optimizers.

8. **`ui.py`** — interactive Panel/Bokeh calibration GUI (`gui` extra). `CalibratedDataset`
   pairs a dataset with a model; `CoordinateCorrectionLayout` wires sliders/tables to live
   re-projection so a user can tune calibration against a displayed dataset.

### Mental model for making changes

- The central invariant is that `trace`/model code must stay valid under **all three backends**
  (float, sympy, jax). Branching on a value (`if x > 0`) breaks the jax tracer — that is why
  you see `try/except TracerBoolConversionError` and `_one`/`flip_factor` style tricks
  (constants threaded through so operations stay differentiable instead of using Python
  conditionals). Preserve that pattern.
- Respect the `PixelYX` vs `CoordXY` distinction; mixing axis order is the most likely source
  of silent geometric bugs.
- JAX x64 is enabled at import time (`jax.config.update("jax_enable_x64", True)`) in every
  module that uses JAX — this must happen before other JAX imports, hence the `# noqa: E702`
  and `# ruff: disable[E402]` markers at the top of those files.

## Testing notes

- `tests/conftest.py` provides a `random_model` fixture (randomized `Model4DSTEM`) used for
  property/round-trip style tests.
- The `with_numba` marker (declared in `pytest.ini`) tags tests that exercise Numba kernels so
  they can be re-run with `NUMBA_DISABLE_JIT=1` for coverage of the interpreted paths.
- CI (`.github/workflows/`) runs on Python 3.14 across Linux/Windows/macOS with all extras
  installed.
