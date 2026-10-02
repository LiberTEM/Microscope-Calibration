'''
Helper functions specific to 4D STEM DM4 files acquired at
JEOL Neo-ARM at CEA in Grenoble.

To be tested and adapted to other microscopes!
'''

from typing import Self
from ncempy.io.dm import fileDM
import pint
import flax
import json
import numpy as np

from microscope_calibration.common.model import Model4DSTEM, DescanError
from microscope_calibration.ui import CalibratedDataset
from libertem.api import Context


class CalibratedJEOLDM4(CalibratedDataset):
    def __init__(self, path: str, model: Model4DSTEM, ctx: Context | None = None):
        handle = fileDM(path)
        self.path = path
        self.tags = handle.allTags
        if ctx is None:
            ctx = Context.make_with('inline')
        super().__init__(dataset=ctx.load('auto', path), model=model)

    @classmethod
    def new(cls, path: str, model: Model4DSTEM = None, ctx: Context | None = None) -> Self:
        handle = fileDM(path)
        model = derive_model_from_dm4(handle, model)
        return cls(path=path, model=model, ctx=ctx)

    def derive_relative_for(self, path: str, ctx: Context | None = None) -> Self:
        old_handle = fileDM(self.path)
        new_handle = fileDM(path)
        model = derive_model_relative_dm4(
            old_dm4=old_handle,
            new_dm4=new_handle,
            model=self.model
        )
        return self.__class__(
            path=path,
            model=model,
            ctx=ctx,
        )

    @property
    def dm4(self) -> fileDM:
        return fileDM(self.path)

    def save(self, path: str):
        statedict = flax.serialization.to_state_dict(
            (self.path, self.model.normalize_types())
        )
        with open(path, mode='w') as f:
            json.dump(statedict, f)

    @classmethod
    def load(cls, path: str, ctx: Context | None = None) -> Self:
        with open(path) as f:
            statedict = json.load(f)
        dm4path, model = flax.serialization.from_state_dict(
            target=('', Model4DSTEM.default()),
            state=statedict
        )
        return cls(path=dm4path, model=model, ctx=ctx)

    def microscope_info(self) -> dict:
        tags = self.dm4.allTags
        result = {}
        search = 'ImageTags.Microscope Info.'
        for tag, value in tags.items():
            if search in tag:
                subtag = tag.rsplit(search, 1)[-1]
                if isinstance(value, np.number):
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    if len(value.shape) != 1:
                        raise RuntimeError("Only works for 1D shapes for now")
                    value = tuple(it. item() for it in value)
                result[subtag] = value
        return result

    def compare(self, other: "CalibratedJEOLDM4") -> dict:
        old_info = self.microscope_info()
        new_info = other.microscope_info()
        tmp_for_keys = old_info.copy()
        tmp_for_keys.update(new_info)

        dm4 = {}
        for key in tmp_for_keys:
            old_val = old_info.get(key)
            new_val = new_info.get(key)
            if old_val != new_val:
                dm4[key] = (old_val, new_val)

        model_attrs = list(Model4DSTEM.__dataclass_fields__.keys())
        model_attrs.remove('descan_error')

        old_model = self.model
        new_model = other.model

        model = {}

        for attr in model_attrs:
            old_val = getattr(old_model, attr)
            new_val = getattr(new_model, attr)
            if not np.allclose(old_val, new_val, rtol=1e-5, atol=1e-15):
                model[attr] = (old_val, new_val)

        descan_error_attrs = list(DescanError.__annotations__.keys())
        descan_error = {}
        for attr in descan_error_attrs:
            old_val = getattr(old_model.descan_error, attr)
            new_val = getattr(new_model.descan_error, attr)
            if not np.allclose(old_val, new_val, rtol=1e-5, atol=1e-15):
                descan_error[attr] = (old_val, new_val)

        return {
            'dm4': dm4,
            'model': model,
            'descan_error': descan_error,
        }

    def calibrated(self, model: Model4DSTEM, ctx: Context | None = None) -> "CalibratedJEOLDM4":
        return self.__class__(
            path=self.path,
            model=model,
            ctx=ctx
        )


def acceleration_from_dm4(dm4file: fileDM) -> pint.Quantity:
    tags = dm4file.allTags

    acceleration_voltage_V = tags['.ImageList.2.ImageTags.Microscope Info.Voltage']
    return pint.Quantity(acceleration_voltage_V, 'V')


def derive_model_from_dm4(dm4file: fileDM, model: Model4DSTEM | None = None) -> Model4DSTEM:
    tags = dm4file.allTags
    # 4D STEM DM4 was traditionally transposed so order of sig and nav is swapped
    shape_keys = [
        # This is nav!
        '.ImageList.2.ImageData.Dimensions.3',
        '.ImageList.2.ImageData.Dimensions.4',
        # this is sig!
        '.ImageList.2.ImageData.Dimensions.1',
        '.ImageList.2.ImageData.Dimensions.2',
    ]
    shape = tuple(int(tags[key]) for key in shape_keys)

    if model is None:
        model = Model4DSTEM.default(dataset_shape=shape)

    scan_step_y_number = tags['.ImageList.2.ImageData.Calibrations.Dimension.3.Scale']
    scan_step_y_unit = tags['.ImageList.2.ImageData.Calibrations.Dimension.3.Units']
    scan_step_y = pint.Quantity(scan_step_y_number, scan_step_y_unit)

    scan_step_x_number = tags['.ImageList.2.ImageData.Calibrations.Dimension.4.Scale']
    scan_step_x_unit = tags['.ImageList.2.ImageData.Calibrations.Dimension.4.Units']
    scan_step_x = pint.Quantity(scan_step_x_number, scan_step_x_unit)
    if scan_step_x != scan_step_y:
        raise ValueError("Requires uniform scan step in Y and X for the time being")

    # Note camera raw pixel pitch vs effective one (binning!)
    cam_pixel_pitches = tags['.ImageList.2.ImageTags.Acquisition.Frame.CCD.Pixel Size (um)']
    if len(cam_pixel_pitches) > 2:
        raise ValueError("Camera pixel pitch should have two dimensions")
    if len(cam_pixel_pitches) == 2 and cam_pixel_pitches[0] != cam_pixel_pitches[1]:
        raise ValueError("Requires uniform pixel pitch in Y and X for the time being")
    cam_pixel_pitch = pint.Quantity(cam_pixel_pitches[0], 'um')

    camera_length = pint.Quantity(
        tags['.ImageList.2.ImageTags.Microscope Info.STEM Camera Length'],
        'mm'
    )
    # Seems to be opposite of what Model4DSTEM works with
    scan_rotation = pint.Quantity(
        -tags['.ImageList.2.ImageTags.DigiScan.Rotation'],
        'degree'
    )
    return model.derive(
        scan_pixel_pitch=scan_step_x.to('m').magnitude,
        scan_rotation=scan_rotation.to('radian').magnitude,
        camera_length=camera_length.to('m').magnitude,
        detector_pixel_pitch=cam_pixel_pitch.to('m').magnitude
    )


def derive_model_relative_dm4(old_dm4: fileDM, new_dm4: fileDM, model: Model4DSTEM) -> Model4DSTEM:
    old_valmodel = derive_model_from_dm4(old_dm4)
    new_valmodel = derive_model_from_dm4(new_dm4)

    old_tags = old_dm4.allTags
    new_tags = new_dm4.allTags

    # assume overfocus is adjusted with stage
    # TODO also calibrate focus length scale?
    stage_z_key = '.ImageList.2.ImageTags.Microscope Info.Stage Position.Stage Z'
    old_z = pint.Quantity(old_tags[stage_z_key], 'um')
    new_z = pint.Quantity(new_tags[stage_z_key], 'um')

    return model.derive(
        scan_pixel_pitch=(
            model.scan_pixel_pitch * new_valmodel.scan_pixel_pitch / old_valmodel.scan_pixel_pitch
        ),
        detector_pixel_pitch=(
            model.detector_pixel_pitch
            * new_valmodel.detector_pixel_pitch / old_valmodel.detector_pixel_pitch
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
        overfocus=model.overfocus - (new_z - old_z).to('m').magnitude
    )
