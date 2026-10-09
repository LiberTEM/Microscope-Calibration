# Microscope-Calibration

Tools to calibrate a (scanning) transmission electron microscope from data, with a focus on
4D-STEM. The package recovers the geometric calibration of a 4D-STEM experiment — scan and
detector rotation, handedness (flip), camera length, scan pixel pitch, overfocus and descan
error — and uses it to correct and reconstruct datasets.

The scientific background is described in the preprint https://arxiv.org/abs/2403.08538.

**Status:** alpha, under active development. The API may change without notice.

## Overview

- **`Model4DSTEM`** (`microscope_calibration.common.model`) holds the full geometric calibration
  of a 4D-STEM experiment and traces rays through it using
  [TemGymCore](https://github.com/TemGym/TemGymCore). The same model works with plain floats,
  SymPy symbols (exact symbolic solutions) and JAX (automatic differentiation, JIT), so it can
  be used to reason about ray paths and to build solvers and optimizers.
- **Optimizers** (`microscope_calibration.util.optimize`) fit calibration parameters such as
  camera length, scan pixel pitch and descan error to reference data.
- **LiberTEM UDFs** (`microscope_calibration.udf`) apply a calibration to real datasets, e.g. to
  reconstruct overfocused 4D-STEM data.
- **Simulator** (`microscope_calibration.util.stem_overfocus_sim`) generates synthetic
  overfocused 4D-STEM data from a model.
- **Interactive GUI** (`microscope_calibration.ui`) to adjust a calibration against a displayed
  dataset, optionally with diffraction rings computed from a CIF crystal structure.

## Installation

Requires Python >= 3.14. The package is not on PyPI yet. Install it from GitHub with all
optional features:

```bash
pip install "microscope-calibration[common,diffraction,gui] @ git+https://github.com/LiberTEM/Microscope-Calibration"
```

To install a specific release, append `@<tag>` to the URL.

The core dependencies only cover `Model4DSTEM`. Optional extras:

- `common`: LiberTEM, Numba, SciPy, optimizers etc. for UDFs, simulator and fitting
- `diffraction`: crystallography support to compute diffraction angles from CIF files
- `gui`: the interactive calibration interface
- `test`: test dependencies

## Examples

The notebooks in [`examples/`](examples/) are currently the best documentation:

- [`model.ipynb`](examples/model.ipynb): basic use of `Model4DSTEM` to reason about ray paths,
  with exact symbolic (SymPy) and numerical (JAX + Optimistix) solutions.
- [`generate.ipynb`](examples/generate.ipynb): simulate an overfocused 4D-STEM dataset.
- [`stem_overfocus.ipynb`](examples/stem_overfocus.ipynb): interactive calibration of a series
  of real datasets, including descan error, camera length using diffraction rings, and
  overfocus. The data will be published on Zenodo.

## Development

Install from a clone of the repository in editable mode:

```bash
git clone https://github.com/LiberTEM/Microscope-Calibration
cd Microscope-Calibration
pip install -e .[test,common,diffraction,gui]
pytest tests/
pre-commit run --all-files
```

## Citation

If you use this software, please cite it via its Zenodo record (DOI to be added on first
release) and the preprint https://arxiv.org/abs/2403.08538.

<!-- TODO add Zenodo concept DOI and badge after the first release -->

## License

GPL-3.0, see [LICENSE](LICENSE).

## Previous version

An earlier, much simpler version of this package accompanied the preprint
https://arxiv.org/abs/2403.08538 and was deposited at https://doi.org/10.5281/zenodo.10418769.
The current code is a substantial rewrite and is not compatible with that version.

Changes since that deposition, before the rewrite:

- Fixed definition of camera length in the simulator to match the figure in
  https://arxiv.org/pdf/2403.08538.pdf, PR
  https://github.com/LiberTEM/Microscope-Calibration/pull/17. Previously, the camera length was
  defined from the focus point, not the specimen plane. See also
  https://github.com/TemGym/TemGym/pull/33 for the corresponding update in TemGym. Note that
  the TemGym model used for calculation was correct, only the alternative manual ray tracing
  implementation was affected.
