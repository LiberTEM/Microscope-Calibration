import concurrent.futures
from typing import Literal, Any
from collections.abc import Mapping
import html

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
# ruff: disable[E402]
import flax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import panel as pn
import sparse
from bokeh.events import DoubleTap, Tap
from bokeh.plotting import ColumnDataSource
from libertem.api import Context
from libertem.io.dataset.base import DataSet
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.raw import PickUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem_blobfinder.common.patterns import BackgroundSubtraction
from libertem_blobfinder.udf.correlation import FastCorrelationUDF, run_fastcorrelation
from libertem_ui.display.cursor import Cursor
from libertem_ui.display.lines import Curve
from libertem_ui.display.points import PointSet, RingSet
from libertem_ui.figure import ApertureFigure

from .common.model import DescanError, Model4DSTEM, PixelYX, lambdify_trace_for

# Also available from this module
from .dataset import AbstractCalibratedDataset, CalibratedDataset  # noqa: F401
from .common.stem_overfocus import (
    corrected_det_x,
    corrected_det_y,
    get_center,
    get_detector_correction_matrix,
    ring_radii,
)
from .udf.stem_overfocus import CorrectedPickUDF, OverfocusUDF
from .util.optimize import (
    make_overfocus_loss_function,
    optimize,
    solve_coords_points,
    solve_hit_specimen,
    solve_tilt_descan_error,
    solve_tilt_descan_error_points,
)
# ruff: enable[E402]

NavModeT = Literal[
    "point",
    "sumsig",
]
SigModeT = Literal[
    "lin",
    "log",
]


trace = lambdify_trace_for(jnp)


# Instructions in the layout, see CoordinateCorrectionLayout.layout
SECTION_1_TEXT = """
# 4D STEM coordinate system calibration

**Dataset**: {dataset}

{tags}

This tool helps to check and adjust the geometry of a 4D STEM
experiment.

## De-scan error, convergence semi-angle and effective camera length

<details>
<summary>Instructions</summary>

First, enter the nominal values for the scan rotation as well as the
detector pixel pitch in the corresponding input fields.

Second, confirm or calibrate the shift of the beam center as a
function of scan position (de-scan error).

1. Change the selected scan position in the plot "Uncorrected scan map".
1. Observe if the cursor in the plot "Currently picked uncorrected
   detector frame" stays at the beam center.
1. Observe if the plot "Point analysis corrected for de-scan error"
   shows a bright field image of the specimen.
1. Observe if the plots "Frame corrected for de-scan error" and "Sum
   of frames corrected for de-scan error" show the primary beam
   exactly at the cursor position.
1. If the positions don't match or the plot is distorted, move the
   cursor to the beam center and record the position. Do this for at
   least three different scan positions.
1. Press the button "Apply correction from table" to re-calibrate
   the de-scan error.
1. Observe if the de-scan error is calibrated correctly now by
   repeating the first three steps.

In case the frames show diffraction rings or spots and a list of
"twotheta" angles was provided, now calibrate the camera length so
that the ring glyphs match the peaks or rings in the detector
frames.

Finally, adjust the convergence semi-angle so that the innermost
glyph matches the size of the diffraction disk.

</details>
"""

SCAN_STEP_TEXT = """
## Scan step calibration

<details>

<summary>Instructions</summary>

Move the yellow line handles in the plot "Point analysis corrected for
de-scan error" above so that the line spans a feature of known physical size. Check the
size in the "Scale bar" input field and adjust if necessary. Note
that for defocused data the scale is only correct if the de-scan
error was compensated correctly.

The scale bar angle value is for information only.

</details>
"""

SECTION_2_TEXT = """
## Detector rotation, overfocus and handedness

<details>

<summary>Instructions</summary>

These parameters can only be calibrated in a dataset that was
recorded with a strong defocus so that the detector shows a shadow
image projection of the specimen.

1. Move the green cursor in the plot "Point analysis corrected for
   de-scan error" above to a prominent feature.
1. Select a scan position in the plot "Uncorrected scan map" below
   where this feature is also visible on a detector frame. Selecting
   the position of the green cursor will show the feature at the beam
   center.
1. Observe if the green cursor in the plot "Currently picked
   uncorrected detector frame" below points to the same feature.
1. Observe if the plot "Frames superimposed in scan coordinate
   system" shows a sharp image of the specimen that matches the plot
   "Point analysis corrected for de-scan error".
1. If the positions don't match or the plot is distorted, move the
   green cursor in the plot "Currently picked uncorrected detector
   frame" below to the feature and record the position. Do this for at least three
   different features or scan positions.
1. Press the button "Derive coordinate system from table" to
   re-calibrate the coordinate system.
1. Observe if the coordinate system is calibrated correctly now by
   repeating the first four steps.

</details>
"""


def tags_markdown(tags: Mapping[str, Any]) -> str:
    """
    Collapsible HTML table of the metadata tags of a dataset
    """
    if not tags:
        return "Embedded metadata tags: None"
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in tags.items()
    )
    return (
        "\n<details>\n<summary>Embedded metadata tags</summary>\n"
        f"<table>{rows}</table>\n</details>\n"
    )


def format_model(model: Model4DSTEM, columns: list[str]) -> dict[str, str]:
    """
    Model parameters as text for the model table, rotations in degrees
    """
    res = {}
    for key in columns:
        value = getattr(model, key)
        if isinstance(value, float):
            if "rotation" in key:
                value = np.rad2deg(value)
            value = f"{value:.5}"
            if "rotation" in key:
                value = value + "°"
        else:
            value = str(value)
        res[key] = value
    return res


def format_descan_error(descan_error: DescanError) -> dict[str, str]:
    """
    Descan error components as text for the descan error table
    """
    return {
        key: f"{value:.5}" if isinstance(value, float) else str(value)
        for key, value in descan_error._asdict().items()
    }


def correlation_pattern(ri: np.ndarray, ro: np.ndarray) -> BackgroundSubtraction:
    """
    Pattern to find the primary beam with the ring radii of :func:`ring_radii`
    """
    radius_outer = 1.2 * ro[0]
    if len(ri) >= 2:
        radius_outer = max(radius_outer, ri[1] / 2)
    # Make sure we stay away from other peaks
    return BackgroundSubtraction(radius=ro[0], radius_outer=radius_outer)


def expected_center_shifts(model: Model4DSTEM, nav_shape: tuple[int, int]) -> np.ndarray:
    """
    Shift of the primary beam from the detector center at each scan position
    according to the descan error of :code:`model`, in detector pixels (y, x)
    """
    # Correction to same parameters, except descan error of 0
    ref_model = model.derive(descan_error=DescanError())
    mat = get_detector_correction_matrix(rec_model=model, ref_model=ref_model)
    # Scan positions in the dataset
    y, x = np.mgrid[: nav_shape[0], : nav_shape[1]]
    # Calculate where the nominal center actually is on the detector
    expected_x = corrected_det_x(
        det_corr_x=model.detector_center.x,
        det_corr_y=model.detector_center.y,
        scan_x=x,
        scan_y=y,
        mat=mat,
    )
    expected_y = corrected_det_y(
        det_corr_x=model.detector_center.x,
        det_corr_y=model.detector_center.y,
        scan_x=x,
        scan_y=y,
        mat=mat,
    )
    # Subtract the expected position so that only the deviation remains
    dc = model.detector_center
    return np.stack((expected_y - dc.y, expected_x - dc.x), axis=-1)


def center_regression(field: np.ndarray, center: PixelYX) -> np.ndarray:
    """
    Linear regression of beam positions, following the CoMUDF regression

    Parameters
    ----------
    field
        Beam position (y, x) on the detector for each scan position (y, x)
    center
        Nominal detector center

    Returns
    -------
    regression
        Rows offset, d/dy, d/dx of the deviation from :code:`center`, with
        columns (y, x), see :data:`~microscope_calibration.util.optimize.CoMRegression`
    """
    # Only keep deviation from expected value in regression
    field = field.copy()
    field[..., 0] -= center.y
    field[..., 1] -= center.x
    inp = np.ones(field.shape[:-1] + (3,))
    y, x = np.ogrid[: field.shape[0], : field.shape[1]]
    inp[..., 1] = y
    inp[..., 2] = x
    reg_res = np.linalg.lstsq(inp.reshape((-1, 3)), field.reshape((-1, 2)), rcond=None)
    return reg_res[0]


class CoordinateCorrectionLayout:
    descan_columns = ["scan_y", "scan_x", "detector_cy", "detector_cx"]
    descan_index_cols = descan_columns[:2]

    coord_columns = [
        "scan_y",
        "scan_x",
        "specimen_y",
        "specimen_x",
        "detector_y",
        "detector_x",
    ]
    coord_index_cols = coord_columns[:4]

    model_columns = list(Model4DSTEM.__dataclass_fields__.keys())
    model_columns.remove("descan_error")

    descan_error_columns = list(DescanError.__annotations__.keys())

    def __init__(
        self,
        calibrated_dataset: AbstractCalibratedDataset,
        ctx: Context,
        nav_mode: NavModeT = "point",
        sig_mode: SigModeT = "lin",
        twothetas: np.ndarray | None = None,
    ):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.calibrated_dataset = calibrated_dataset
        self.ctx = ctx
        self.nav_mode = nav_mode
        self.sig_mode = sig_mode

        self.start_model = calibrated_dataset.model

        if twothetas is None:
            twothetas = np.array((0.0,))

        self.coords_adjusted = False

        self._init_state(twothetas)
        self._init_figures()
        self._init_widgets()
        self._connect_events()

    def _init_state(self, twothetas: np.ndarray):
        """
        ColumnDataSources and tables that hold the GUI state
        """
        self.model_model = ColumnDataSource(
            data=(
                self.model_update(self.start_model)
                | self.scan_pos_update(self.start_model.scan_center)
                | self.twothetas_update(twothetas)
            ),
        )

        self.scalebar_model = ColumnDataSource(
            data=self.scale_update(
                start=PixelYX(
                    y=self.start_model.scan_center.y * 1.7,
                    x=self.start_model.scan_center.x * 0.3,
                ),
                stop=PixelYX(
                    y=self.start_model.scan_center.y * 1.7,
                    x=self.start_model.scan_center.x * 1.7,
                ),
            ),
        )

        self.ring_model = ColumnDataSource(data=self.ring_update(self.model_model.data))
        self.centered_ring_model = ColumnDataSource(
            data=self.ring_update(
                self.model_model.data,
                detector_center=self.start_model.detector_center,
            )
        )
        self.descan_fixpoints = pd.DataFrame(columns=self.descan_columns).set_index(
            self.descan_index_cols
        )
        # Don't use scan center because the glyphs cannot be moved separately
        feature_start = PixelYX(
            y=self.start_model.scan_center.y + 5,
            x=self.start_model.scan_center.x + 5,
        )
        self.feature_model = ColumnDataSource(data=self.scan_pos_update(feature_start))
        if self.start_model.overfocus != 0:
            feature_select_start = self.get_feature_on_detector(
                model=self.start_model,
                scan_pos=self.start_model.scan_center,
                specimen_px=feature_start,
            )
        else:
            feature_select_start = self.start_model.detector_center
        self.feature_select_model = ColumnDataSource(data=self.feature_update(feature_select_start))

        self.coord_fixpoints = pd.DataFrame(columns=self.coord_columns).set_index(
            self.coord_index_cols
        )

    def _init_figures(self):
        """
        Figures with cursors and other glyphs, in the order they are added
        """
        preview = self.get_preview()
        self.nav_fig = self._new_figure(
            preview["nav"], title=f"Uncorrected scan map (method: {self.nav_mode})", maxdim=350
        )
        self.cursor = self._add_cursor(self.nav_fig, self.model_model, "scan_x", "scan_y", "red")

        frames = self.get_frames(pos=self.scan_pos)
        self.pick_fig = self._new_figure(
            frames["raw"],
            title=f"Currently picked uncorrected detector frame ({self.sig_mode})",
            maxdim=350,
        )
        self.beam_centre = self._add_cursor(
            self.pick_fig, self.ring_model, "detector_cx", "detector_cy", "red"
        )
        self.diffraction_ringsets = self._add_rings(self.pick_fig, self.ring_model)

        self.corr_point_fig = self._new_figure(
            preview["corrected_point"],
            title="Point analysis corrected for de-scan error",
            maxdim=250,
        )
        self.cursor_2 = self._add_cursor(
            self.corr_point_fig, self.model_model, "scan_x", "scan_y", "red"
        )
        self.nav_feature_cursor = self._add_cursor(
            self.corr_point_fig, self.feature_model, "scan_x", "scan_y", "lightgreen"
        )
        self.scalebar_handles = (
            PointSet(cds=self.scalebar_model, x="scalebar_x", y="scalebar_y")
            .on(self.corr_point_fig.fig)
            .editable(drag=True, tag_name="cursor")
        )
        self.scalebar_line = Curve(
            cds=self.scalebar_model, xkey="scalebar_x", ykey="scalebar_y"
        ).on(self.corr_point_fig.fig)
        self.scalebar_handles.glyph.line_color = "yellow"
        self.scalebar_handles.glyph.fill_color = None
        self.scalebar_line.glyph.line_color = "yellow"

        self.corr_pick_fig = self._new_figure(
            frames["corrected"],
            title=f"Frame corrected for de-scan error ({self.sig_mode})",
            maxdim=250,
        )
        self.corr_beam_centre = self._add_cursor(
            self.corr_pick_fig,
            self.centered_ring_model,
            "detector_cx",
            "detector_cy",
            "red",
            editable=False,
        )
        self.corr_diffraction_ringsets = self._add_rings(
            self.corr_pick_fig, self.centered_ring_model
        )

        self.corr_sum_fig = self._new_figure(
            preview["corrected_sum"],
            title=f"Sum of frames corrected for de-scan error ({self.sig_mode})",
            maxdim=250,
        )
        self.corr_beam_centre_2 = self._add_cursor(
            self.corr_sum_fig,
            self.centered_ring_model,
            "detector_cx",
            "detector_cy",
            "red",
            editable=False,
        )
        self.corr_diffraction_ringsets_2 = self._add_rings(
            self.corr_sum_fig, self.centered_ring_model
        )

        self.nav_fig_2 = self._new_figure(
            preview["nav"], title=f"Uncorrected scan map (method: {self.nav_mode})"
        )
        self.cursor_3 = self._add_cursor(
            self.nav_fig_2, self.model_model, "scan_x", "scan_y", "red"
        )

        self.pick_fig_2 = self._new_figure(
            frames["raw"],
            title=f"Currently picked uncorrected detector frame ({self.sig_mode})",
            maxdim=350,
        )
        self.nav_feature_select_cursor = self._add_cursor(
            self.pick_fig_2, self.feature_select_model, "detector_x", "detector_y", "lightgreen"
        )

        self.back_sum_fig = self._new_figure(
            preview["backprojected_sum"],
            title="Frames superimposed in scan coordinate system",
            maxdim=350,
        )
        self.back_pick_fig = self._new_figure(
            frames["backprojected"],
            title="Currently selected frame back-projected to scan coordinate system",
            maxdim=350,
        )

    @classmethod
    def _new_figure(cls, data: np.ndarray, title: str, maxdim: int | None = None) -> ApertureFigure:
        kwargs = {} if maxdim is None else {"maxdim": maxdim}
        fig = ApertureFigure.new(data, title=title, **kwargs)
        cls.adjust_layout(fig, shape=data.shape)
        return fig

    @staticmethod
    def _add_cursor(
        fig: ApertureFigure,
        cds: ColumnDataSource,
        x: str,
        y: str,
        color: str,
        editable: bool = True,
    ) -> Cursor:
        cursor = Cursor(cds=cds, x=x, y=y).on(fig.fig)
        if editable:
            # this adds the necessary components to make the cursor draggable
            cursor = cursor.editable(selected=True)
        cursor.glyph.line_color = color
        return cursor

    @staticmethod
    def _add_rings(fig: ApertureFigure, cds: ColumnDataSource) -> list[RingSet]:
        """
        Diffraction rings with the radii :code:`ri_{i}` and :code:`ro_{i}` of
        :meth:`ring_update`
        """
        return [
            RingSet(
                cds=cds,
                x="detector_cx",
                y="detector_cy",
                inner_radius=f"ri_{i}",
                outer_radius=f"ro_{i}",
            ).on(fig.fig)
            for i in range(cds.data["num_rings"][0])
        ]

    def _init_widgets(self):
        """
        Inputs, buttons and tables
        """
        self.cl_input = pn.widgets.FloatInput(
            label="Camera length / m", step=0.01, value=self.start_model.camera_length
        )
        self.semiconv_input = pn.widgets.FloatInput(
            label="Convergence semi-angle / mrad",
            step=0.01,
            # The model uses the slope, see Model4DSTEM
            value=float(np.arctan(self.start_model.semiconv)) * 1000,
        )
        self.scalebar_input = pn.widgets.FloatInput(
            label="Scale bar / nm",
            step=0.01,
            value=(
                self.start_model.scan_pixel_pitch
                * self.get_scalebar_length(self.scalebar_model.data)
                * 1e9
            ),
        )
        self.scale_angle_display = pn.widgets.FloatInput(
            label="Scale bar angle / deg",
            disabled=True,
            value=(np.rad2deg(self.get_scalebar_angle(self.scalebar_model.data))),
        )
        self.scan_rotation_input = pn.widgets.FloatInput(
            label="Scan rotation / degrees",
            step=0.1,
            value=self.start_model.scan_rotation * 180 / np.pi,
        )
        self.detector_pitch_input = pn.widgets.FloatInput(
            label="Detector pixel pitch / µm",
            step=0.1,
            value=self.start_model.detector_pixel_pitch * 1e6,
        )

        self.descan_fixpoint_table = self._new_fixpoint_table(self.descan_fixpoints)

        self.record_button = pn.widgets.Button(label="Record")
        self.apply_button = pn.widgets.Button(label="Apply correction from table")
        self.clear_button = pn.widgets.Button(label="Clear")
        self.correlate_button = pn.widgets.Button(label="Refine correction with cross-correlation")

        self.coord_fixpoint_table = self._new_fixpoint_table(self.coord_fixpoints)

        self.coord_record_button = pn.widgets.Button(label="Record")
        self.coord_apply_button = pn.widgets.Button(label="Derive coordinate system from table")
        self.coord_clear_button = pn.widgets.Button(label="Clear")
        self.optimize_button = pn.widgets.Button(label="Optimize sharpness of back-projection")

        self.invert_focus_button = pn.widgets.Button(label="Invert sign of overfocus")

        self.model_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=self.model_columns),
            # See https://github.com/holoviz/panel/pull/8256
            selectable=False,
        )
        self.descan_error_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=self.descan_error_columns),
            # See https://github.com/holoviz/panel/pull/8256
            selectable=False,
        )
        self.update_model_tables(self.start_model)

    @staticmethod
    def _new_fixpoint_table(df: pd.DataFrame) -> pn.widgets.Tabulator:
        return pn.widgets.Tabulator(
            df,
            buttons={
                "select": '<i class="fa fa-bullseye"></i>',
                "delete": '<i class="fa fa-trash"></i>',
            },
            # See https://github.com/holoviz/panel/pull/8256
            selectable=False,
        )

    def _connect_events(self):
        # ## state
        # trigger whenever the "data" attribute of the cursor ColumnDataSource changes
        self.model_model.on_change("data", self.on_model_model_change)
        self.scalebar_model.on_change("data", self.on_scalebar_model_change)
        self.feature_model.on_change("data", self.on_feature_model_change)

        # ## nav_fig
        self.nav_fig.fig.on_event(Tap, self.move_scan_to)
        self.corr_point_fig.fig.on_event(Tap, self.move_feature_to)
        self.nav_fig_2.fig.on_event(Tap, self.move_scan_to)

        # ## pick_fig
        self.pick_fig.fig.on_event(Tap, self.move_rings_to)
        self.pick_fig.fig.on_event(DoubleTap, self.move_rings_to)
        self.pick_fig.fig.on_event(DoubleTap, lambda e: self.descan_add_row())

        self.pick_fig_2.fig.on_event(Tap, self.move_feature_select_to)
        self.pick_fig_2.fig.on_event(DoubleTap, self.move_feature_select_to)
        self.pick_fig_2.fig.on_event(DoubleTap, lambda e: self.coord_add_row())

        # ## Model param controls
        self.cl_input.param.watch(self.update_cl, "value")
        self.semiconv_input.param.watch(self.update_semiconv, "value")
        self.scalebar_input.param.watch(self.update_scale, "value")
        self.detector_pitch_input.param.watch(self.update_detector_pitch, "value")
        self.scan_rotation_input.param.watch(self.update_scan_rotation, "value")

        # ## descan fixpoints table
        self.descan_fixpoint_table.on_click(
            lambda e: self.descan_delete_row(e.row), column="delete"
        )
        self.descan_fixpoint_table.on_click(
            lambda e: self.descan_move_to_row(e.row), column="select"
        )

        # ## descan correction buttons
        self.record_button.on_click(lambda e: self.descan_add_row())
        self.apply_button.on_click(lambda e: self.perform_descan_update())
        self.clear_button.on_click(lambda e: self.descan_drop())
        self.correlate_button.on_click(lambda e: self.center_correlation_regression())

        # ## coord fixpoints table
        self.coord_fixpoint_table.on_click(lambda e: self.coord_delete_row(e.row), column="delete")
        self.coord_fixpoint_table.on_click(lambda e: self.coord_move_to_row(e.row), column="select")

        # ## coordinate system buttons
        self.coord_record_button.on_click(lambda e: self.coord_add_row())
        self.coord_apply_button.on_click(lambda e: self.perform_coord_update())
        self.coord_clear_button.on_click(lambda e: self.coord_drop())
        self.optimize_button.on_click(lambda e: self.sharpen())
        self.invert_focus_button.on_click(lambda e: self.invert_focus())

    @staticmethod
    def adjust_layout(plot, shape):
        plot.fig.y_range.bounds = (0, shape[0])
        plot.fig.x_range.bounds = (0, shape[1])

    @property
    def dataset(self) -> DataSet:
        return self.calibrated_dataset.dataset

    def current_calibrated(self) -> AbstractCalibratedDataset:
        return self.calibrated_dataset.calibrated(self.model)

    def update_model_tables(self, model):
        model_df = self.model_table.value
        for key, value in format_model(model, self.model_columns).items():
            model_df.loc[0, key] = value
        self.model_table.value = model_df

        descan_df = self.descan_error_table.value
        for key, value in format_descan_error(model.descan_error).items():
            descan_df.loc[0, key] = value
        self.descan_error_table.value = descan_df

    def on_model_model_change(self, attr, old, new):
        # This is a "bokeh-style" callback because it will trigger directly from a ColumnDataSource
        # The callback must have three arguments [attr, old, new]
        # attr will be "data"
        # old will be the CDS dict before the trigger
        # new will be the CDS dict after the trigger
        assert attr == "data"
        new_model = self.get_model(new)
        # Re-calibrating the coordinate system or updating only what changed
        # doesn't work, likely because not all changes trigger this callback,
        # see https://github.com/holoviz/panel/issues/8275. Update everything.
        new_pos = self.get_scan_pos(new)
        frames = self.get_frames(pos=new_pos)
        self.pick_fig.update(frames["raw"])
        self.pick_fig_2.update(frames["raw"])
        self.corr_pick_fig.update(frames["corrected"])
        self.back_pick_fig.update(frames["backprojected"])

        previews = self.get_preview()
        self.corr_point_fig.update(previews["corrected_point"])
        self.corr_sum_fig.update(previews["corrected_sum"])
        self.back_sum_fig.update(previews["backprojected_sum"])

        self.update_with_force(
            self.ring_model,
            self.ring_update(model_data=new),
        )
        self.update_with_force(
            self.centered_ring_model,
            self.ring_update(model_data=new, detector_center=new_model.detector_center),
        )
        detector_pos = self.get_feature_on_detector(
            model=new_model,
            scan_pos=new_pos,
            specimen_px=self.get_scan_pos(self.feature_model.data),
        )
        self.update_with_force(self.feature_select_model, self.feature_update(detector_pos))
        self.update_model_tables(new_model)

    def on_scalebar_model_change(self, attr, old, new):
        # This is a "bokeh-style" callback because it will trigger directly from a ColumnDataSource
        # The callback must have three arguments [attr, old, new]
        # attr will be "data"
        # old will be the CDS dict before the trigger
        # new will be the CDS dict after the trigger
        assert attr == "data"
        # For some reason we can't get change events from the scale bar CDS,
        # but receive model_model CDS changes
        length = self.get_scalebar_length(self.scalebar_model.data)
        # TODO use microscope_calibration.util.optimize.solve_scan_pixel_pitch
        # instead to not make assumption about linearity
        self.scalebar_input.value = length * self.model.scan_pixel_pitch * 1e9
        self.scale_angle_display.value = np.rad2deg(self.get_scalebar_angle(new))
        self.push()

    def on_feature_model_change(self, attr, old, new):
        # This is a "bokeh-style" callback because it will trigger directly from a ColumnDataSource
        # The callback must have three arguments [attr, old, new]
        # attr will be "data"
        # old will be the CDS dict before the trigger
        # new will be the CDS dict after the trigger
        assert attr == "data"
        model = self.model
        if model.overfocus != 0:
            feature_pos = self.get_feature_on_detector(
                model=model,
                scan_pos=self.get_scan_pos(self.model_model.data),
                specimen_px=self.get_scan_pos(new),
            )
            self.update_with_force(self.feature_select_model, self.feature_update(feature_pos))

    def update_cl(self, event):
        base = self.model
        model = base.adjust_camera_length(camera_length=event.new)
        self.update_with_force(self.model_model, self.model_update(model))

    def update_semiconv(self, event):
        model = self.model.derive(semiconv=float(np.tan(event.new / 1000)))
        self.update_with_force(self.model_model, self.model_update(model))

    def update_scale(self, event):
        base = self.model
        # TODO use microscope_calibration.util.optimize.solve_scan_pixel_pitch
        # instead to not make assumption about linearity
        length_px = self.get_scalebar_length(self.scalebar_model.data)
        scan_pixel_pitch = event.new / 1e9 / length_px
        model = base.adjust_scan_pixel_pitch(scan_pixel_pitch)
        self.update_with_force(self.model_model, self.model_update(model))

    def update_scan_rotation(self, event):
        model = self.model.adjust_scan_rotation(scan_rotation=event.new / 180 * np.pi)
        self.update_with_force(self.model_model, self.model_update(model))

    def update_detector_pitch(self, event):
        model = self.model.adjust_detector_pixel_pitch(detector_pixel_pitch=event.new / 1e6)
        self.update_with_force(self.model_model, self.model_update(model))

    def move_scan_to(self, event):
        self.update_with_force(
            self.model_model,
            self.scan_pos_update(
                PixelYX(y=float(event.y), x=float(event.x)),
            ),
        )

    def move_feature_to(self, event):
        self.update_with_force(
            self.feature_model,
            self.scan_pos_update(
                PixelYX(y=float(event.y), x=float(event.x)),
            ),
        )

    def move_feature_select_to(self, event):
        self.update_with_force(
            self.feature_select_model,
            self.feature_update(
                PixelYX(y=float(event.y), x=float(event.x)),
            ),
        )

    def move_rings_to(self, event):
        self.update_with_force(
            self.ring_model,
            self.detector_center_update(
                PixelYX(y=float(event.y), x=float(event.x)),
            ),
        )

    @staticmethod
    def _upsert_row(table: pn.widgets.Tabulator, columns: list, index_cols: list, values: tuple):
        """
        Add a row to a fix point table, or update it if the index already exists
        """
        idx = values[: len(index_cols)]
        new_df = pd.DataFrame([values], columns=columns).set_index(index_cols)
        if idx in table.value.index:
            table.value.update(new_df)
            # Make sure the change is registered
            table.value = table.value
        else:
            table.value = pd.concat((table.value, new_df))

    @staticmethod
    def _delete_row(table: pn.widgets.Tabulator, row: int):
        df = table.value
        table.value = df.drop(index=df.index[row])

    @staticmethod
    def _drop_rows(table: pn.widgets.Tabulator):
        table.value = table.value.drop(table.value.index)

    def descan_add_row(self):
        scan_pos = self.scan_pos
        center_pos = self.get_detector_center(self.ring_model.data)
        self._upsert_row(
            self.descan_fixpoint_table,
            self.descan_columns,
            self.descan_index_cols,
            (int(np.round(scan_pos.y)), int(np.round(scan_pos.x)), center_pos.y, center_pos.x),
        )

    def descan_delete_row(self, row):
        self._delete_row(self.descan_fixpoint_table, row)

    def descan_move_to_row(self, row):
        df = self.descan_fixpoint_table.value
        key = df.index[row]
        assert len(key) == 2
        scan_pos = PixelYX(y=key[0], x=key[1])
        self.update_with_force(self.model_model, self.scan_pos_update(scan_pos))

    def descan_drop(self):
        self._drop_rows(self.descan_fixpoint_table)

    def coord_add_row(self):
        scan_pos = self.scan_pos
        feature_nav_pos = self.get_scan_pos(self.feature_model.data)
        feature_pos = self.get_feature(self.feature_select_model.data)
        self._upsert_row(
            self.coord_fixpoint_table,
            self.coord_columns,
            self.coord_index_cols,
            (
                int(np.round(scan_pos.y)),
                int(np.round(scan_pos.x)),
                float(feature_nav_pos.y),
                float(feature_nav_pos.x),
                feature_pos.y,
                feature_pos.x,
            ),
        )
        self.coords_adjusted = False

    def coord_delete_row(self, row):
        self._delete_row(self.coord_fixpoint_table, row)
        self.coords_adjusted = False

    def coord_move_to_row(self, row):
        df = self.coord_fixpoint_table.value
        key = df.index[row]
        detector = df.values[row]
        assert len(key) == 4
        scan_pos = PixelYX(y=key[0], x=key[1])
        specimen_pos = PixelYX(y=key[2], x=key[3])
        self.update_with_force(self.model_model, self.scan_pos_update(scan_pos))
        self.update_with_force(self.feature_model, self.scan_pos_update(specimen_pos))
        detector_pos = PixelYX(y=detector[0], x=detector[1])
        self.update_with_force(self.feature_select_model, self.feature_update(detector_pos))

    def coord_drop(self):
        self._drop_rows(self.coord_fixpoint_table)
        self.coords_adjusted = False

    def get_model(self, model_data) -> Model4DSTEM:
        return self.deserialize(model_data["model"][0])

    @staticmethod
    def get_scan_pos(model_data):
        return PixelYX(y=model_data["scan_y"][0], x=model_data["scan_x"][0])

    @staticmethod
    def get_model_twothetas(model_data):
        return model_data["twothetas"][0]

    @staticmethod
    def get_detector_center(ring_data):
        return PixelYX(
            y=ring_data["detector_cy"][0],
            x=ring_data["detector_cx"][0],
        )

    @staticmethod
    def get_feature(feature_select_data):
        return PixelYX(
            y=feature_select_data["detector_y"][0],
            x=feature_select_data["detector_x"][0],
        )

    @staticmethod
    def get_scalebar_length(scalebar_data):
        start = np.array((scalebar_data["scalebar_y"][0], scalebar_data["scalebar_x"][0]))
        stop = np.array((scalebar_data["scalebar_y"][1], scalebar_data["scalebar_x"][1]))
        return np.linalg.norm(stop - start)

    @staticmethod
    def get_scalebar_angle(scalebar_data) -> float:
        start = np.array((scalebar_data["scalebar_y"][0], scalebar_data["scalebar_x"][0]))
        stop = np.array((scalebar_data["scalebar_y"][1], scalebar_data["scalebar_x"][1]))
        vector = stop - start
        return float(np.arctan2(*vector))

    @staticmethod
    def get_feature_on_detector(
        model: Model4DSTEM, scan_pos: PixelYX, specimen_px: PixelYX
    ) -> PixelYX:
        slope, residual = solve_hit_specimen(
            model=model,
            scan_pos=scan_pos,
            specimen_px=specimen_px,
        )
        res = trace(
            model,
            scan_pos=scan_pos,
            source_dy=slope.dy,
            source_dx=slope.dx,
        )
        return res["detector"].sampling["detector_px"]

    @property
    def model(self) -> Model4DSTEM:
        return self.get_model(self.model_model.data)

    @property
    def scan_pos(self):
        return self.get_scan_pos(self.model_model.data)

    def get_preview(self):
        if self.nav_mode == "sumsig":
            nav_udf = SumSigUDF()
        elif self.nav_mode == "point":
            cy = self.model.detector_center.y
            cx = self.model.detector_center.x
            sig_shape = tuple(self.dataset.shape.sig)

            def get_mask():
                a = sparse.COO(
                    data=np.array([1]),
                    coords=np.array(([int(cy)], [int(cx)])),
                    shape=sig_shape,
                )
                return a

            nav_udf = ApplyMasksUDF(
                mask_factories=[get_mask],
                use_sparse=True,
            )
        else:
            raise NotImplementedError()
        overfocus_udf = OverfocusUDF(overfocus_model={"model": self.model})

        res = self.ctx.run_udf(dataset=self.dataset, udf=(nav_udf, overfocus_udf))
        result = {}
        if self.nav_mode == "sumsig":
            result["nav"] = res[0]["intensity"].data
            result["corrected_point"] = res[0]["intensity"].data
        elif self.nav_mode == "point":
            result["nav"] = res[0]["intensity"].data[..., 0]
            result["corrected_point"] = res[1]["corrected_point"].data
        result["backprojected_sum"] = res[1]["backprojected_sum"].data
        if self.sig_mode == "lin":
            result["corrected_sum"] = res[1]["corrected_sum"].data
        elif self.sig_mode == "log":
            result["corrected_sum"] = np.log1p(np.maximum(0, res[1]["corrected_sum"].data))
        else:
            raise ValueError()
        return result

    def get_frames(self, pos: PixelYX):
        y = int(round(pos.y))
        x = int(round(pos.x))
        roi = self.dataset.roi[y, x]
        udfs = (
            PickUDF(),
            CorrectedPickUDF(overfocus_model={"model": self.model}),
        )
        res = self.ctx.run_udf(dataset=self.dataset, udf=udfs, roi=roi)

        if self.sig_mode == "log":

            def fn(data):
                return np.log1p(np.maximum(0, data))
        else:

            def fn(data):
                return data

        return {
            "raw": fn(res[0]["intensity"].raw_data[0]),
            "corrected": fn(res[1]["corrected"].raw_data[0]),
            "backprojected": fn(res[1]["backprojected"].raw_data[0]),
        }

    @staticmethod
    def serialize(model: Model4DSTEM):
        return flax.serialization.to_state_dict(model.normalize_types())

    def deserialize(self, state_dict):
        res = flax.serialization.from_state_dict(
            target=self.start_model, state=state_dict
        ).normalize_types()
        return res

    def model_update(self, model: Model4DSTEM):
        return {"model": [self.serialize(model)]}

    @staticmethod
    def scan_pos_update(scan_pos: PixelYX):
        return {
            "scan_y": [float(scan_pos.y)],
            "scan_x": [float(scan_pos.x)],
        }

    @staticmethod
    def twothetas_update(twothetas):
        return {
            "twothetas": [twothetas],
        }

    @staticmethod
    def scale_update(start: PixelYX, stop: PixelYX):
        return {
            "scalebar_x": [float(start.x), float(stop.x)],
            "scalebar_y": [float(start.y), float(stop.y)],
        }

    def ring_update(self, model_data, detector_center=None):
        model = self.get_model(model_data)
        ri, ro = ring_radii(model=model, twothetas=self.get_model_twothetas(model_data))
        if detector_center is None:
            detector_center = get_center(
                model=model,
                scan_pos=self.get_scan_pos(model_data),
            )
        return (
            {
                "detector_cy": [float(detector_center.y)],
                "detector_cx": [float(detector_center.x)],
                "num_rings": [len(ri)],
            }
            | {f"ri_{i}": [float(rii)] for i, rii in enumerate(ri)}
            | {f"ro_{i}": [float(roo)] for i, roo in enumerate(ro)}
        )

    def feature_update(self, detector_pos: PixelYX):
        return {
            "detector_y": [float(detector_pos.y)],
            "detector_x": [float(detector_pos.x)],
        }

    @staticmethod
    def detector_center_update(detector_center: PixelYX):
        return {
            "detector_cy": [float(detector_center.y)],
            "detector_cx": [float(detector_center.x)],
        }

    def perform_descan_update(self):
        points = self.descan_fixpoint_table.value.reset_index().to_numpy(dtype=np.float64)
        if len(points):
            new_model, residual = solve_tilt_descan_error_points(
                ref_model=self.model, points=jnp.array(points)
            )
            self.update_with_force(self.model_model, self.model_update(new_model))

    def perform_coord_update(self):
        points = self.coord_fixpoint_table.value.reset_index().to_numpy(dtype=np.float64)
        if len(points):
            new_model, residual = solve_coords_points(
                ref_model=self.model, points=jnp.array(points)
            )
            self.update_with_force(self.model_model, self.model_update(new_model))
            self.coords_adjusted = True

    def invert_focus(self):
        self.update_with_force(self.model_model, self.model_update(self.model.invert_focus()))

    def _push(self):
        self.nav_fig.push(
            self.pick_fig,
            self.corr_point_fig,
            self.corr_pick_fig,
            self.corr_sum_fig,
            self.nav_fig_2,
            self.pick_fig_2,
            self.back_sum_fig,
            self.back_pick_fig,
        )

    def push(self):
        self._push()
        self.executor.submit(self._push)

    def update_with_force(self, cds: ColumnDataSource, update):
        cds.data.update(**update)
        self.push()
        if cds.document is not None:
            holding = cds.document.callbacks._hold
            cds.document.callbacks.unhold()
        else:
            holding = None
        cds.data.update(**update)
        self.push()
        if cds.document is not None:
            cds.document.callbacks.hold(holding)

    def center_correlation_regression(self):
        model = self.model
        ri, ro = ring_radii(model, self.get_model_twothetas(self.model_model.data))
        shifts = expected_center_shifts(model, nav_shape=tuple(self.dataset.shape.nav))
        aux_shifts = FastCorrelationUDF.aux_data(
            data=shifts,
            kind="nav",
            extra_shape=(2,),
            dtype=shifts.dtype,
        )
        res = run_fastcorrelation(
            ctx=self.ctx,
            dataset=self.dataset,
            peaks=np.array([model.detector_center]),
            match_pattern=correlation_pattern(ri, ro),
            zero_shift=aux_shifts,
            upsample=True,
        )
        regression = center_regression(
            res["refineds"].data[:, :, 0, :], center=model.detector_center
        )
        new_model, residual = solve_tilt_descan_error(ref_model=model, regression=regression)
        self.model_model.data.update(**self.model_update(new_model))
        self.push()

    def sharpen(self):
        def callback(args, model, res, blur):
            self.back_sum_fig.update(res[0]["backprojected_sum"].data)
            self.update_model_tables(model)

        make_new_model, loss = make_overfocus_loss_function(
            model=self.model,
            ctx=self.ctx,
            dataset=self.dataset,
            overfocus_udf=OverfocusUDF(overfocus_model={"model": self.model}),
            callback=callback,
        )

        res = optimize(loss)
        model = make_new_model(res.x)
        self.update_with_force(self.model_model, self.model_update(model))

    @property
    def layout(self):
        self.section_1_label = pn.pane.Markdown(
            SECTION_1_TEXT.format(
                dataset=html.escape(str(self.calibrated_dataset.dataset)),
                tags=tags_markdown(self.calibrated_dataset.tags),
            ),
            max_width=500,
        )
        inputs_1 = pn.layout.Row(
            self.scan_rotation_input,
            self.detector_pitch_input,
            self.cl_input,
            self.semiconv_input,
        )
        self.scan_step_label = pn.pane.Markdown(SCAN_STEP_TEXT)
        inputs_2 = pn.layout.Row(
            self.scalebar_input,
            self.scale_angle_display,
        )
        raw_figs = pn.layout.Row(self.nav_fig.layout, self.pick_fig.layout)
        corrected_figs = pn.layout.Row(
            self.corr_point_fig.layout,
            self.corr_pick_fig.layout,
            self.corr_sum_fig.layout,
        )

        self.section_2_label = pn.pane.Markdown(SECTION_2_TEXT, max_width=500)
        raw_figs_2 = pn.layout.Row(self.nav_fig_2.layout, self.pick_fig_2.layout)
        back_figs = pn.layout.Row(self.back_sum_fig.layout, self.back_pick_fig.layout)
        descan_buttons = pn.layout.Row(
            self.record_button,
            self.apply_button,
            self.clear_button,
            self.correlate_button,
        )
        self.descan_label = pn.pane.Markdown(
            "### De-scan correction table",
        )
        descan_section = pn.layout.Column(
            self.descan_label, self.descan_fixpoint_table, descan_buttons
        )
        coord_buttons = pn.layout.Row(
            self.coord_record_button,
            self.coord_apply_button,
            self.coord_clear_button,
            self.optimize_button,
        )
        self.coord_label = pn.pane.Markdown(
            "### Coordinate system calibration table",
        )
        coord_section = pn.layout.Column(self.coord_label, self.coord_fixpoint_table, coord_buttons)
        self.result_label = pn.pane.Markdown(
            "### Current model parameters",
        )
        result_section = pn.layout.Column(
            self.result_label,
            self.invert_focus_button,
            self.model_table,
            self.descan_error_table,
        )
        return pn.layout.Column(
            self.section_1_label,
            inputs_1,
            raw_figs,
            corrected_figs,
            descan_section,
            self.scan_step_label,
            inputs_2,
            self.section_2_label,
            raw_figs_2,
            coord_section,
            back_figs,
            result_section,
        )
