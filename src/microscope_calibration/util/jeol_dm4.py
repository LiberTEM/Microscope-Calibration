'''
Helper functions specific to 4D STEM DM4 files acquired at
JEOL Neo-ARM at CEA in Grenoble.

To be tested and adapted to other microscopes!
'''

from ncempy.io.dm import fileDM
import pint

from microscope_calibration.common.model import Model4DSTEM


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
        # FIXME check direction
        # TODO also include focus
        overfocus=model.overfocus + (new_z - old_z).to('m').magnitude
    )
