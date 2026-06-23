from numpy.testing import assert_allclose

import numpy as np
import jax
from libertem.api import Context

from microscope_calibration.common.stem_overfocus import (
    get_backward_transformation_matrix, get_detector_correction_matrix,
    project_frame_backwards, correct_frame
)
from microscope_calibration.udf.stem_overfocus import OverfocusUDF
from microscope_calibration.common.model import (
    Parameters4DSTEM, PixelYX, DescanError, trace
)


@jax.jit
def get_beam_center(params: Parameters4DSTEM, scan_y, scan_x):
    res = trace(
        params=params,
        scan_pos=PixelYX(y=scan_y, x=scan_x),
        source_dx=0.,
        source_dy=0.
    )
    center = res['detector'].sampling['detector_px']
    return (center.y, center.x)


def test_udf():
    params = Parameters4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.234,
        camera_length=0.73,
        detector_pixel_pitch=0.0321,
        semiconv=0.023,
        scan_center=PixelYX(x=0.13, y=0.23),
        scan_rotation=0.752,
        flip_factor=-1.,
        detector_center=PixelYX(x=5, y=7),
        detector_rotation=2.134,
        descan_error=DescanError(
            pxo_pxi=0.2,
            pxo_pyi=0.3,
            pyo_pxi=0.5,
            pyo_pyi=0.7,
            sxo_pxi=0.11,
            sxo_pyi=0.13,
            syo_pxi=0.17,
            syo_pyi=0.19,
            offpxi=0.23,
            offpyi=0.29,
            offsxi=0.31,
            offsyi=0.37
        )
    )
    back_mat = get_backward_transformation_matrix(rec_params=params)
    corr_mat = get_detector_correction_matrix(rec_params=params)

    data = np.random.random((9, 11, 13, 17))

    ctx = Context.make_with('inline')
    ds = ctx.load('memory', data=data)

    ref_back = np.zeros_like(data[:, :, 0, 0])
    ref_corr = np.zeros_like(data[0, 0])
    ref_point = np.zeros_like(data[:, :, 0, 0])

    for scan_y in range(ds.shape.nav[0]):
        for scan_x in range(ds.shape.nav[1]):
            (y, x) = get_beam_center(params=params, scan_y=scan_y, scan_x=scan_x)
            y = int(np.round(y))
            x = int(np.round(x))
            if y >= 0 and y < data.shape[2] and x >= 0 and x < data.shape[3]:
                ref_point[scan_y, scan_x] = data[
                    scan_y, scan_x, y, x
                ]
    ref_select = np.zeros_like(ref_back)

    select_y = data.shape[0]//2
    select_x = data.shape[1]//2

    for scan_y in range(data.shape[0]):
        for scan_x in range(data.shape[1]):
            project_frame_backwards(
                frame=data[scan_y, scan_x],
                source_semiconv=params.semiconv,
                mat=back_mat,
                scan_y=scan_y,
                scan_x=scan_x,
                image_out=ref_back,
            )
            if select_y == scan_y and select_x == scan_x:
                project_frame_backwards(
                    frame=data[scan_y, scan_x],
                    source_semiconv=params.semiconv,
                    mat=back_mat,
                    scan_y=scan_y,
                    scan_x=scan_x,
                    image_out=ref_select,
                )
            correct_frame(
                frame=data[scan_y, scan_x],
                mat=corr_mat,
                scan_y=scan_y,
                scan_x=scan_x,
                detector_out=ref_corr,
            )

    res = ctx.run_udf(dataset=ds, udf=OverfocusUDF(overfocus_params={'params': params}))

    assert_allclose(ref_back, res['backprojected_sum'])
    assert_allclose(ref_corr, res['corrected_sum'])
    assert_allclose(ref_point, res['corrected_point'])
