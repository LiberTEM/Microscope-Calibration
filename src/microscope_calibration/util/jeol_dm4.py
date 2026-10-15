"""
Helper functions specific to 4D STEM DM4 files acquired at
JEOL Neo-ARM at CEA in Grenoble.

To be tested and adapted to other microscopes!
"""

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, Self

import flax.serialization
import numpy as np
import pint
from libertem.api import Context
from ncempy.io.dm import fileDM

from microscope_calibration.common.model import DescanError, Model4DSTEM
from microscope_calibration.dataset import CalibratedDataset

# Tags of the 4D STEM data. ImageList.1 is the thumbnail.
IMAGE = ".ImageList.2"

DM4Tags = Mapping[str, Any]


def read_tags(path: str) -> DM4Tags:
    """
    All tags of a DM4 file, see :attr:`ncempy.io.dm.fileDM.allTags`
    """
    with fileDM(path) as handle:
        return handle.allTags


class CalibratedJEOLDM4(CalibratedDataset):
    def __init__(self, path: str, model: Model4DSTEM, ctx: Context | None = None):
        self.path = path
        self._tags = read_tags(path)
        if ctx is None:
            ctx = Context.make_with("inline")
        super().__init__(dataset=ctx.load("auto", path), model=model)

    @classmethod
    def new(cls, path: str, model: Model4DSTEM | None = None, ctx: Context | None = None) -> Self:
        model = derive_model_from_dm4(read_tags(path), model)
        return cls(path=path, model=model, ctx=ctx)

    def derive_relative_for(self, path: str, ctx: Context | None = None) -> Self:
        model = derive_model_relative_dm4(
            old_tags=self.tags, new_tags=read_tags(path), model=self.model
        )
        return self.__class__(
            path=path,
            model=model,
            ctx=ctx,
        )

    def save(self, path: str):
        statedict = flax.serialization.to_state_dict((self.path, self.model.normalize_types()))
        with open(path, mode="w") as f:
            json.dump(statedict, f)

    @classmethod
    def load(cls, path: str, ctx: Context | None = None) -> Self:
        with open(path) as f:
            statedict = json.load(f)
        dm4path, model = flax.serialization.from_state_dict(
            target=("", Model4DSTEM.default()), state=statedict
        )
        return cls(path=dm4path, model=model, ctx=ctx)  # pyright: ignore[reportArgumentType]  # flax types the result too loosely

    @property
    def tags(self) -> DM4Tags:
        return self._tags

    def microscope_info(self) -> dict:
        return microscope_info(self.tags)

    def compare(self, other: CalibratedJEOLDM4) -> dict:
        return compare(
            old_tags=self.tags, old_model=self.model, new_tags=other.tags, new_model=other.model
        )

    def calibrated(self, model: Model4DSTEM, ctx: Context | None = None) -> CalibratedJEOLDM4:
        return self.__class__(path=self.path, model=model, ctx=ctx)


def microscope_info(tags: DM4Tags) -> dict:
    """
    Microscope Info tags of the 4D STEM data, with keys relative to it
    """
    result = {}
    search = f"{IMAGE}.ImageTags.Microscope Info."
    for tag, value in tags.items():
        if tag.startswith(search):
            subtag = tag.removeprefix(search)
            if isinstance(value, np.number):
                value = value.item()
            elif isinstance(value, np.ndarray):
                if len(value.shape) != 1:
                    raise RuntimeError("Only works for 1D shapes for now")
                value = tuple(it.item() for it in value)
            result[subtag] = value
    return result


def compare(
    old_tags: DM4Tags, old_model: Model4DSTEM, new_tags: DM4Tags, new_model: Model4DSTEM
) -> dict:
    """
    Differences of microscope info, model parameters and descan error

    Returns
    -------
    differences
        :code:`{"dm4": ..., "model": ..., "descan_error": ...}`, each a dict
        of :code:`(old, new)` values for the keys that differ
    """
    old_info = microscope_info(old_tags)
    new_info = microscope_info(new_tags)
    dm4 = {}
    for key in old_info | new_info:
        old_val = old_info.get(key)
        new_val = new_info.get(key)
        if old_val != new_val:
            dm4[key] = (old_val, new_val)

    model = {}
    for field in dataclasses.fields(Model4DSTEM):
        if field.name == "descan_error":
            continue
        old_val = getattr(old_model, field.name)
        new_val = getattr(new_model, field.name)
        if not np.allclose(old_val, new_val, rtol=1e-5, atol=1e-15):
            model[field.name] = (old_val, new_val)

    descan_error = {}
    for attr in DescanError._fields:
        old_val = getattr(old_model.descan_error, attr)
        new_val = getattr(new_model.descan_error, attr)
        if not np.allclose(old_val, new_val, rtol=1e-5, atol=1e-15):
            descan_error[attr] = (old_val, new_val)

    return {
        "dm4": dm4,
        "model": model,
        "descan_error": descan_error,
    }


def acceleration_from_dm4(tags: DM4Tags) -> pint.Quantity:
    """
    Acceleration voltage, see :func:`read_tags`
    """
    acceleration_voltage_V = tags[f"{IMAGE}.ImageTags.Microscope Info.Voltage"]
    return pint.Quantity(acceleration_voltage_V, "V")


def derive_model_from_dm4(tags: DM4Tags, model: Model4DSTEM | None = None) -> Model4DSTEM:
    """
    Model with scan and detector calibration from DM4 tags, see :func:`read_tags`

    Parameters
    ----------
    tags
        Tags of a DM4 file
    model
        Model to start from. By default, :meth:`Model4DSTEM.default` for the
        dataset shape.
    """
    # 4D STEM DM4 was traditionally transposed so order of sig and nav is swapped
    shape_keys = [
        # This is nav!
        f"{IMAGE}.ImageData.Dimensions.3",
        f"{IMAGE}.ImageData.Dimensions.4",
        # this is sig!
        f"{IMAGE}.ImageData.Dimensions.1",
        f"{IMAGE}.ImageData.Dimensions.2",
    ]
    shape = tuple(int(tags[key]) for key in shape_keys)

    if model is None:
        model = Model4DSTEM.default(dataset_shape=shape)

    scan_step_y_number = tags[f"{IMAGE}.ImageData.Calibrations.Dimension.3.Scale"]
    scan_step_y_unit = tags[f"{IMAGE}.ImageData.Calibrations.Dimension.3.Units"]
    scan_step_y = pint.Quantity(scan_step_y_number, scan_step_y_unit)

    scan_step_x_number = tags[f"{IMAGE}.ImageData.Calibrations.Dimension.4.Scale"]
    scan_step_x_unit = tags[f"{IMAGE}.ImageData.Calibrations.Dimension.4.Units"]
    scan_step_x = pint.Quantity(scan_step_x_number, scan_step_x_unit)
    if scan_step_x != scan_step_y:
        raise ValueError("Requires uniform scan step in Y and X for the time being")

    # Note camera raw pixel pitch vs effective one (binning!)
    cam_pixel_pitches = tags[f"{IMAGE}.ImageTags.Acquisition.Frame.CCD.Pixel Size (um)"]
    if len(cam_pixel_pitches) > 2:
        raise ValueError("Camera pixel pitch should have two dimensions")
    if len(cam_pixel_pitches) == 2 and cam_pixel_pitches[0] != cam_pixel_pitches[1]:
        raise ValueError("Requires uniform pixel pitch in Y and X for the time being")
    cam_pixel_pitch = pint.Quantity(cam_pixel_pitches[0], "um")

    camera_length = pint.Quantity(
        tags[f"{IMAGE}.ImageTags.Microscope Info.STEM Camera Length"], "mm"
    )
    # Seems to be opposite of what Model4DSTEM works with
    scan_rotation = pint.Quantity(-tags[f"{IMAGE}.ImageTags.DigiScan.Rotation"], "degree")
    # Seems to be opposite of what Model4DSTEM works with
    detector_rotation = pint.Quantity(
        -tags[f"{IMAGE}.ImageTags.DigiScan.Rotation Offset"], "degree"
    )
    return model.derive(
        scan_pixel_pitch=scan_step_x.to("m").magnitude,
        scan_rotation=scan_rotation.to("radian").magnitude,
        detector_rotation=detector_rotation.to("radian").magnitude,
        camera_length=camera_length.to("m").magnitude,
        detector_pixel_pitch=cam_pixel_pitch.to("m").magnitude,
    )


def derive_model_relative_dm4(
    old_tags: DM4Tags, new_tags: DM4Tags, model: Model4DSTEM
) -> Model4DSTEM:
    """
    Apply the changes between two DM4 files to a model calibrated for the old one

    Scan and detector pixel pitch and camera length are scaled, scan rotation
    and overfocus (from the stage Z position) are shifted by the differences.
    """
    old_valmodel = derive_model_from_dm4(old_tags)
    new_valmodel = derive_model_from_dm4(new_tags)

    # assume overfocus is adjusted with stage
    # TODO also calibrate focus length scale?
    stage_z_key = f"{IMAGE}.ImageTags.Microscope Info.Stage Position.Stage Z"
    old_z = pint.Quantity(old_tags[stage_z_key], "um")
    new_z = pint.Quantity(new_tags[stage_z_key], "um")

    return model.derive(
        scan_pixel_pitch=(
            model.scan_pixel_pitch * new_valmodel.scan_pixel_pitch / old_valmodel.scan_pixel_pitch
        ),
        detector_pixel_pitch=(
            model.detector_pixel_pitch
            * new_valmodel.detector_pixel_pitch
            / old_valmodel.detector_pixel_pitch
        ),
        camera_length=(
            model.camera_length * new_valmodel.camera_length / old_valmodel.camera_length
        ),
        scan_rotation=(
            model.scan_rotation + new_valmodel.scan_rotation - old_valmodel.scan_rotation
        ),
        # Stage Z points up, meaning at constant focus
        # it reduces overfocus
        # TODO also include focus
        overfocus=model.overfocus - (new_z - old_z).to("m").magnitude,
    )
