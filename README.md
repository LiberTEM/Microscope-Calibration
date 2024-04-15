# Microscope-Calibration
Tools to calibrate a microscope

## Rotation and handedness

See preprint at https://arxiv.org/abs/2403.08538 for a description and https://github.com/LiberTEM/Microscope-Calibration/blob/main/examples/stem_overfocus.ipynb for an example.

### Changelog

### Since deposition https://doi.org/10.5281/zenodo.10418769

* Fix definition of camera length in the simulator to match figure in
  https://arxiv.org/pdf/2403.08538.pdf, PR
  https://github.com/LiberTEM/Microscope-Calibration/pull/17. Previously, the
  camera length was defined from the focus point, not the specimen plane. See
  also https://github.com/TemGym/TemGym/pull/33 for the corresponding update in
  TemGym. Note that the TemGym model used for calculation was correct, only the
  alternative manual ray tracing implementation was affected.
