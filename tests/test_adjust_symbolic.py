import itertools

import pytest
import sympy as sym
from temgym_core import CoordXY, PixelYX
from temgym_core.components import DescanError

from microscope_calibration.common.model import Model4DSTEM, flip_y, rotate
from microscope_calibration.util.sympy import symbol_maker

# The descan error compensation of Model4DSTEM.adjust_*() is derived
# symbolically from Model4DSTEM.trace(). The functions below are the previous
# hand-written implementations. They serve as a reference to confirm that the
# derivation reduces to the known analytical formulas.


def reference_adjust_scan_rotation(self, scan_rotation: float):
    """
    Adjust the scan rotation while keeping the effective descan error
    compensation the same.

    This allows first compensating descan error and then adjusting other parameters.
    """
    de = self.descan_error
    angle = scan_rotation - self.scan_rotation
    # Rotate the input direction

    def trans(c: CoordXY):
        return rotate(c=c, radians=angle)

    pxo_pyx = trans(CoordXY(y=de.pxo_pyi, x=de.pxo_pxi))
    pyo_pyx = trans(CoordXY(y=de.pyo_pyi, x=de.pyo_pxi))
    sxo_pyx = trans(CoordXY(y=de.sxo_pyi, x=de.sxo_pxi))
    syo_pyx = trans(CoordXY(y=de.syo_pyi, x=de.syo_pxi))
    new_de = DescanError(
        pxo_pyi=pxo_pyx.y,
        pyo_pyi=pyo_pyx.y,
        pxo_pxi=pxo_pyx.x,
        pyo_pxi=pyo_pyx.x,
        sxo_pyi=sxo_pyx.y,
        syo_pyi=syo_pyx.y,
        sxo_pxi=sxo_pyx.x,
        syo_pxi=syo_pyx.x,
        offpxi=de.offpxi,
        offpyi=de.offpyi,
        offsxi=de.offsxi,
        offsyi=de.offsyi,
    )
    return self.derive(
        scan_rotation=scan_rotation,
        descan_error=new_de,
    )


def reference_adjust_scan_pixel_pitch(self, scan_pixel_pitch: float):
    """
    Adjust the scan pixel pitch while keeping the effective descan error
    compensation the same.

    This allows first compensating descan error and then adjusting other parameters.
    """
    de = self.descan_error
    ratio = self.scan_pixel_pitch / scan_pixel_pitch

    new_de = DescanError(
        pxo_pyi=de.pxo_pyi * ratio,
        pyo_pyi=de.pyo_pyi * ratio,
        pxo_pxi=de.pxo_pxi * ratio,
        pyo_pxi=de.pyo_pxi * ratio,
        sxo_pyi=de.sxo_pyi * ratio,
        syo_pyi=de.syo_pyi * ratio,
        sxo_pxi=de.sxo_pxi * ratio,
        syo_pxi=de.syo_pxi * ratio,
        offpxi=de.offpxi,
        offpyi=de.offpyi,
        offsxi=de.offsxi,
        offsyi=de.offsyi,
    )
    return self.derive(
        scan_pixel_pitch=scan_pixel_pitch,
        descan_error=new_de,
    )


def reference_adjust_scan_center(self, scan_center: PixelYX):
    # Compensate effect of different scan centers with
    # constant offsets of the descanner. We simply measure how much these offsets should be
    # by comparing rays along the optical axis
    res1 = self.trace(scan_pos=self.scan_center, source_dx=0.0, source_dy=0.0)
    res2 = self.trace(scan_pos=scan_center, source_dx=0.0, source_dy=0.0)

    de = self.descan_error
    offpxi = de.offpxi + res2["descanner"].ray.x - res1["descanner"].ray.x
    offpyi = de.offpyi + res2["descanner"].ray.y - res1["descanner"].ray.y
    offsxi = de.offsxi + res2["descanner"].ray.dx - res1["descanner"].ray.dx
    offsyi = de.offsyi + res2["descanner"].ray.dy - res1["descanner"].ray.dy

    new_de = DescanError(
        pxo_pyi=de.pxo_pyi,
        pyo_pyi=de.pyo_pyi,
        pxo_pxi=de.pxo_pxi,
        pyo_pxi=de.pyo_pxi,
        sxo_pyi=de.sxo_pyi,
        syo_pyi=de.syo_pyi,
        sxo_pxi=de.sxo_pxi,
        syo_pxi=de.syo_pxi,
        offpxi=offpxi,
        offpyi=offpyi,
        offsxi=offsxi,
        offsyi=offsyi,
    )
    return self.derive(
        scan_center=scan_center,
        descan_error=new_de,
    )


def reference_adjust_detector_rotation(self, detector_rotation: float):
    de = self.descan_error
    angle = detector_rotation - self.detector_rotation
    # rotate the output direction

    def trans(c: CoordXY):
        return rotate(c=c, radians=angle)

    pxy_pyi = trans(CoordXY(y=de.pyo_pyi, x=de.pxo_pyi))
    pxy_pxi = trans(CoordXY(y=de.pyo_pxi, x=de.pxo_pxi))
    sxy_pyi = trans(CoordXY(y=de.syo_pyi, x=de.sxo_pyi))
    sxy_pxi = trans(CoordXY(y=de.syo_pxi, x=de.sxo_pxi))
    offp = trans(CoordXY(y=de.offpyi, x=de.offpxi))
    offs = trans(CoordXY(y=de.offsyi, x=de.offsxi))
    new_de = DescanError(
        pxo_pyi=pxy_pyi.x,
        pyo_pyi=pxy_pyi.y,
        pxo_pxi=pxy_pxi.x,
        pyo_pxi=pxy_pxi.y,
        sxo_pyi=sxy_pyi.x,
        syo_pyi=sxy_pyi.y,
        sxo_pxi=sxy_pxi.x,
        syo_pxi=sxy_pxi.y,
        offpxi=offp.x,
        offpyi=offp.y,
        offsxi=offs.x,
        offsyi=offs.y,
    )

    return self.derive(
        detector_rotation=detector_rotation,
        descan_error=new_de,
    )


def reference_adjust_flip_factor(self, flip_factor: float):
    de = self.descan_error
    angle = self.detector_rotation

    if flip_factor != self.flip_factor:
        # Rotate into detector directions, flip, then rotate back
        def trans(c: CoordXY):
            return rotate(
                flip_y(
                    c=rotate(c=c, radians=-angle),
                    flip_factor=flip_factor / self.flip_factor,
                ),
                radians=angle,
            )

        # transform the output direction
        pxy_pyi = trans(CoordXY(y=de.pyo_pyi, x=de.pxo_pyi))
        pxy_pxi = trans(CoordXY(y=de.pyo_pxi, x=de.pxo_pxi))
        sxy_pyi = trans(CoordXY(y=de.syo_pyi, x=de.sxo_pyi))
        sxy_pxi = trans(CoordXY(y=de.syo_pxi, x=de.sxo_pxi))
        offp = trans(CoordXY(y=de.offpyi, x=de.offpxi))
        offs = trans(CoordXY(y=de.offsyi, x=de.offsxi))
        new_de = DescanError(
            pxo_pyi=pxy_pyi.x,
            pyo_pyi=pxy_pyi.y,
            pxo_pxi=pxy_pxi.x,
            pyo_pxi=pxy_pxi.y,
            sxo_pyi=sxy_pyi.x,
            syo_pyi=sxy_pyi.y,
            sxo_pxi=sxy_pxi.x,
            syo_pxi=sxy_pxi.y,
            offpxi=offp.x,
            offpyi=offp.y,
            offsxi=offs.x,
            offsyi=offs.y,
        )
        return self.derive(
            flip_factor=flip_factor,
            descan_error=new_de,
        )
    return self


def reference_adjust_detector_center(self, detector_center: PixelYX):
    de = self.descan_error
    zero = PixelYX(0, 0)
    other = self.derive(
        detector_center=detector_center,
    )

    physical_1 = self.detector_to_real(zero)
    physical_2 = other.detector_to_real(zero)
    offpyi = de.offpyi + physical_2.y - physical_1.y
    offpxi = de.offpxi + physical_2.x - physical_1.x
    new_de = DescanError(
        pxo_pyi=de.pxo_pyi,
        pyo_pyi=de.pyo_pyi,
        pxo_pxi=de.pxo_pxi,
        pyo_pxi=de.pyo_pxi,
        sxo_pyi=de.sxo_pyi,
        syo_pyi=de.syo_pyi,
        sxo_pxi=de.sxo_pxi,
        syo_pxi=de.syo_pxi,
        offpxi=offpxi,
        offpyi=offpyi,
        offsxi=de.offsxi,
        offsyi=de.offsyi,
    )
    return self.derive(
        detector_center=detector_center,
        descan_error=new_de,
    )


def reference_adjust_detector_pixel_pitch(self, detector_pixel_pitch: float):
    de = self.descan_error
    ratio = detector_pixel_pitch / self.detector_pixel_pitch

    new_de = DescanError(
        pxo_pyi=de.pxo_pyi * ratio,
        pyo_pyi=de.pyo_pyi * ratio,
        pxo_pxi=de.pxo_pxi * ratio,
        pyo_pxi=de.pyo_pxi * ratio,
        sxo_pyi=de.sxo_pyi * ratio,
        syo_pyi=de.syo_pyi * ratio,
        sxo_pxi=de.sxo_pxi * ratio,
        syo_pxi=de.syo_pxi * ratio,
        offpxi=de.offpxi * ratio,
        offpyi=de.offpyi * ratio,
        offsxi=de.offsxi * ratio,
        offsyi=de.offsyi * ratio,
    )
    return self.derive(
        detector_pixel_pitch=detector_pixel_pitch,
        descan_error=new_de,
    )


def reference_adjust_camera_length(self, camera_length: float):
    de = self.descan_error
    ratio = self.camera_length / camera_length

    new_de = DescanError(
        pxo_pyi=de.pxo_pyi,
        pyo_pyi=de.pyo_pyi,
        pxo_pxi=de.pxo_pxi,
        pyo_pxi=de.pyo_pxi,
        sxo_pyi=de.sxo_pyi * ratio,
        syo_pyi=de.syo_pyi * ratio,
        sxo_pxi=de.sxo_pxi * ratio,
        syo_pxi=de.syo_pxi * ratio,
        offpxi=de.offpxi,
        offpyi=de.offpyi,
        offsxi=de.offsxi * ratio,
        offsyi=de.offsyi * ratio,
    )
    return self.derive(
        camera_length=camera_length,
        descan_error=new_de,
    )


REFERENCES = {
    "scan_rotation": reference_adjust_scan_rotation,
    "scan_pixel_pitch": reference_adjust_scan_pixel_pitch,
    "scan_center": reference_adjust_scan_center,
    "detector_rotation": reference_adjust_detector_rotation,
    "flip_factor": reference_adjust_flip_factor,
    "detector_center": reference_adjust_detector_center,
    "detector_pixel_pitch": reference_adjust_detector_pixel_pitch,
    "camera_length": reference_adjust_camera_length,
}


@pytest.mark.parametrize("param", sorted(REFERENCES))
def test_adjust_symbolic(param):
    model = symbol_maker(Model4DSTEM, postfix="old", recurse_for=(PixelYX, DescanError))
    value = symbol_maker(Model4DSTEM.__annotations__[param], postfix="new", recurse_for=(PixelYX,))
    modified = getattr(model, f"adjust_{param}")(value)
    reference = REFERENCES[param](model, value)

    assert modified.derive(descan_error=model.descan_error) == model.derive(**{param: value})

    # The hand-written formulas for flipping assume a flip factor of +-1, e.g.
    # f_new / f_old instead of f_old / f_new. Check all combinations.
    flip_symbols = [model.flip_factor] + ([value] if param == "flip_factor" else [])
    for flips in itertools.product((-1, 1), repeat=len(flip_symbols)):
        subs = dict(zip(flip_symbols, flips, strict=True))
        for name, res, ref in zip(
            DescanError._fields, modified.descan_error, reference.descan_error, strict=True
        ):
            assert sym.simplify(sym.sympify(res - ref).subs(subs)) == 0, (name, flips)
