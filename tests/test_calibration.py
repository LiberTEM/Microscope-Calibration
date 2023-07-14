from libertem.api import Context

from microscope_calibration.udf.stem_overfocus import (
    OverfocusUDF, OverfocusParams, OverfocusProcessParams
)


def test_smoke():
    ctx = Context.make_with('inline')
    ds = ctx.load('memory', datashape=(6, 7, 8, 9))
    udf = OverfocusUDF(
        overfocus_params=OverfocusParams(
            overfocus=0.0001,
            camera_length=0.15,
            scan_pixel_size=1e-6,
            detector_pixel_size=55e-6,
            semiconv=0.02,
            cy=3,
            cx=4,
            flip_y=False,
            scan_rotation=23,
        ),
        process_params=OverfocusProcessParams(
            pair_distance=10,
            pair_radius=1,
        ),
        point_y=1,
        point_x=2,
    )
    ctx.run_udf(dataset=ds, udf=udf)
