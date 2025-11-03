from typing import Literal
import concurrent.futures

import numpy as np
import sparse
import panel as pn
import flax
import jax.numpy as jnp
import pandas as pd

from libertem.io.dataset.base import DataSet
from libertem.api import Context
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.raw import PickUDF

from libertem_blobfinder.udf.correlation import FastCorrelationUDF, run_fastcorrelation
from libertem_blobfinder.common.patterns import BackgroundSubtraction

from libertem_ui.display.points import RingSet
from libertem_ui.figure import ApertureFigure
from libertem_ui.display.cursor import Cursor
from libertem_ui.display.points import PointSet
from libertem_ui.display.lines import Curve

from bokeh.plotting import ColumnDataSource
from bokeh.events import DoubleTap, Tap

from .common.model import Parameters4DSTEM, DescanError, PixelYX
from .util.optimize import solve_tilt_descan_error_points, solve_tilt_descan_error
from .common.stem_overfocus import (
    get_detector_correction_matrix,
    corrected_det_y,
    corrected_det_x,
    ring_radii,
    get_center,
)
from .udf.stem_overfocus import CorrectedPickUDF, OverfocusUDF


NavModeT = Literal["point", "sumsig"]


class CoordinateCorrectionLayout:
    descan_columns = ["scan_y", "scan_x", "detector_cy", "detector_cx"]
    descan_index_cols = descan_columns[:2]

    coord_columns = ["scan_y", "scan_x", "detector_y", "detector_x"]
    coord_index_cols = coord_columns[:2]

    def __init__(
        self,
        dataset: DataSet,
        ctx: Context,
        nav_mode: NavModeT = "point",
        twothetas: np.ndarray | None = None,
        start_params=None,
    ):
        # # Stuff
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.dataset = dataset
        self.ctx = ctx
        self.nav_mode = nav_mode

        if start_params is None:
            start_params = self.default_params()
        self.start_params = start_params

        if twothetas is None:
            twothetas = np.array((0.0,))

        # # GUI state
        self.model_params = ColumnDataSource(
            data=(
                self.params_update(self.start_params)
                | self.scan_pos_update(self.start_params.scan_center)
                | self.twothetas_update(twothetas)
            ),
        )

        self.scalebar_params = ColumnDataSource(
            data=self.scale_update(
                start=PixelYX(
                    y=self.start_params.scan_center.y * 1.7,
                    x=self.start_params.scan_center.x * 0.3,
                ),
                stop=PixelYX(
                    y=self.start_params.scan_center.y * 1.7,
                    x=self.start_params.scan_center.x * 1.7,
                ),
            ),
        )
        self.ring_params = ColumnDataSource(
            data=self.ring_update(self.model_params.data)
        )
        self.descan_fixpoints = pd.DataFrame(columns=self.descan_columns).set_index(
            self.descan_index_cols
        )
        self.coord_fixpoints = pd.DataFrame(columns=self.coord_columns).set_index(
            self.coord_index_cols
        )

        # # GUI elements
        preview = self.get_preview()
        self.nav_fig = ApertureFigure.new(preview['nav'], title="Nav map")
        self.adjust_layout(self.nav_fig, shape=preview['nav'].shape)

        self.cursor = (
            Cursor(cds=self.model_params, x="scan_x", y="scan_y")
            .on(self.nav_fig.fig)
            # this adds the necessary components to make the cursor draggable
            .editable(selected=True)
        )
        # The cursor Glyph is accessed via the .cursor property, until the API is unified
        self.cursor.cursor.line_color = "red"

        frames = self.get_frames(pos=self.scan_pos)
        self.pick_fig = ApertureFigure.new(
            frames['raw'], title="QWERT"
        )
        self.adjust_layout(self.pick_fig, shape=frames['raw'].shape)

        self.beam_centre = (
            Cursor(cds=self.ring_params, x="detector_cx", y="detector_cy")
            .on(self.pick_fig.fig)
            .editable(selected=True)
        )
        self.beam_centre.glyph.line_color = "red"
        self.diffraction_ringsets = []
        for i in range(self.ring_params.data["num_rings"][0]):
            rikey = f"ri_{i}"
            rokey = f"ro_{i}"
            self.diffraction_ringsets.append(
                RingSet(
                    cds=self.ring_params,
                    x="detector_cx",
                    y="detector_cy",
                    inner_radius=rikey,
                    outer_radius=rokey,
                ).on(self.pick_fig.fig)
            )

        self.corr_point_fig = ApertureFigure.new(
            preview['corrected_point'], title="Corrected point analysis"
        )
        self.adjust_layout(self.corr_point_fig, shape=preview['corrected_point'].shape)
        self.cursor_2 = (
            Cursor(cds=self.model_params, x="scan_x", y="scan_y")
            .on(self.corr_point_fig.fig)
            # this adds the necessary components to make the cursor draggable
            .editable(selected=True)
        )
        # The cursor Glyph is accessed via the .cursor property, until the API is unified
        self.cursor_2.cursor.line_color = "red"

        self.scalebar_handles = (
            PointSet(cds=self.scalebar_params, x="scalebar_x", y="scalebar_y")
            .on(self.corr_point_fig.fig)
            .editable(drag=True, tag_name="cursor")
        )
        self.scalebar_line = Curve(
            cds=self.scalebar_params, xkey="scalebar_x", ykey="scalebar_y"
        ).on(self.corr_point_fig.fig)
        self.scalebar_handles.glyph.line_color = "yellow"
        self.scalebar_handles.glyph.fill_color = None
        self.scalebar_line.glyph.line_color = "yellow"

        self.corr_pick_fig = ApertureFigure.new(
            frames['corrected'], title="Frame corrected for descan error"
        )
        self.adjust_layout(self.corr_pick_fig, shape=frames['corrected'].shape)
        self.corr_sum_fig = ApertureFigure.new(
            preview['corrected_sum'], title="Sum of frames corrected for descan error"
        )
        self.adjust_layout(self.corr_sum_fig, shape=preview['corrected_sum'].shape)

        self.back_sum_fig = ApertureFigure.new(
            preview['backprojected_sum'], title="Frames back-projected to scan coordinate system"
        )
        self.adjust_layout(self.back_sum_fig, shape=preview['backprojected_sum'].shape)

        self.back_pick_fig = ApertureFigure.new(
            frames['backprojected'], title="Current frame back-projected to scan coordinate system"
        )
        self.adjust_layout(self.back_pick_fig, shape=frames['backprojected'].shape)

        self.cl_input = pn.widgets.FloatInput(
            name="Camera length / m", step=0.01, value=self.start_params.camera_length
        )
        self.semiconv_input = pn.widgets.FloatInput(
            name="Convergence semi-angle / mrad",
            step=0.01,
            value=start_params.semiconv * 1000,
        )
        self.scalebar_input = pn.widgets.FloatInput(
            name="Scale bar / nm",
            step=0.01,
            value=(
                self.start_params.scan_pixel_pitch
                * self.get_scalebar_length(self.scalebar_params.data)
                * 1e9
            ),
        )

        self.descan_fixpoint_table = pn.widgets.Tabulator(
            self.descan_fixpoints,
            buttons={
                "select": '<i class="fa fa-bullseye"></i>',
                "delete": '<i class="fa fa-trash"></i>',
            },
            # See https://github.com/holoviz/panel/pull/8256
            selectable=False,
        )

        self.record_button = pn.widgets.Button(name="Record")
        self.apply_button = pn.widgets.Button(name="Apply correction from table")
        self.clear_button = pn.widgets.Button(name="Clear")
        self.correlate_button = pn.widgets.Button(
            name="Refine correction with cross-correlation"
        )

        self.coord_fixpoint_table = pn.widgets.Tabulator(
            self.coord_fixpoints,
            buttons={
                "select": '<i class="fa fa-bullseye"></i>',
                "delete": '<i class="fa fa-trash"></i>',
            },
            # See https://github.com/holoviz/panel/pull/8256
            selectable=False,
        )

        self.coord_record_button = pn.widgets.Button(name="Record")
        self.coord_apply_button = pn.widgets.Button(name="Derive coordinate system from table")
        self.coord_clear_button = pn.widgets.Button(name="Clear")
        self.optimize_button = pn.widgets.Button(
            name="Optimize sharpness of back-projection"
        )

        # # Event handler setup

        # ## state
        # trigger whenever the "data" attribute of the cursor ColumnDataSource changes
        self.model_params.on_change("data", self.on_model_params_change)
        # self.model_params.on_change("data", lambda attr, old, new: print(attr, old, new))
        self.scalebar_params.on_change("data", self.on_scalebar_params_change)

        # ## nav_fig
        self.nav_fig.fig.on_event(Tap, self.move_scan_to)

        # ## pick_fig
        self.pick_fig.fig.on_event(Tap, self.move_rings_to)
        self.pick_fig.fig.on_event(DoubleTap, self.move_rings_to)
        self.pick_fig.fig.on_event(DoubleTap, lambda e: self.descan_add_row())

        # ## Model param controls
        self.cl_input.param.watch(self.update_cl, "value")
        self.semiconv_input.param.watch(self.update_semiconv, "value")
        self.scalebar_input.param.watch(self.update_scale, "value")

        # ## descan fixpoints table
        self.descan_fixpoint_table.on_click(
            lambda e: self.descan_delete_row(e.row) if e.column == "delete" else None,
            column="delete",
        )
        self.descan_fixpoint_table.on_click(
            lambda e: self.descan_move_to_row(e.row) if e.column == "select" else None,
            column="select",
        )

        # ## descan correction buttons
        self.record_button.on_click(lambda e: self.descan_add_row())
        self.apply_button.on_click(lambda e: self.perform_descan_update())
        self.clear_button.on_click(lambda e: self.descan_drop())
        self.correlate_button.on_click(lambda e: self.center_correlation_regression())

        # ## coord fixpoints table
        self.coord_fixpoint_table.on_click(
            lambda e: self.coord_delete_row(e.row) if e.column == "delete" else None,
            column="delete",
        )
        self.coord_fixpoint_table.on_click(
            lambda e: self.coord_move_to_row(e.row) if e.column == "select" else None,
            column="select",
        )

        # ## descan correction buttons
        self.coord_record_button.on_click(lambda e: self.coord_add_row())
        self.coord_apply_button.on_click(lambda e: self.perform_coord_update())
        self.coord_clear_button.on_click(lambda e: self.coord_drop())
        self.optimize_button.on_click(lambda e: self.sharpen())

    @staticmethod
    def adjust_layout(plot, shape):
        plot.fig.y_range.bounds = (0, shape[0])
        plot.fig.x_range.bounds = (0, shape[1])
        plot.fig.sizing_mode = 'scale_width'
        plot.layout.max_width = 400
        plot.layout.sizing_mode = 'stretch_width'

    def on_model_params_change(self, attr, old, new):
        # This is a "bokeh-style" callback because it will trigger directly from a ColumnDataSource
        # The callback must have three arguments [attr, old, new]
        # attr will be "data"
        # old will be the CDS dict before the trigger
        # new will be the CDS dict after the trigger
        assert attr == "data"
        new_pos = self.get_scan_pos(new)
        # Somehow doesn't work with all the event handling confusion
        # Just update always and be done with it *sigh*
        # if old_pos != new_pos:
        frames = self.get_frames(pos=new_pos)
        self.pick_fig.update(frames['raw'])
        self.corr_pick_fig.update(frames['corrected'])
        self.back_pick_fig.update(frames['backprojected'])

        previews = self.get_preview()
        self.corr_point_fig.update(previews['corrected_point'])
        self.corr_sum_fig.update(previews['corrected_sum'])
        self.back_sum_fig.update(previews['backprojected_sum'])

        self.update_with_force(
            self.ring_params,
            self.ring_update(model_data=new),
        )
        self.push()

    def on_scalebar_params_change(self, attr, old, new):
        # This is a "bokeh-style" callback because it will trigger directly from a ColumnDataSource
        # The callback must have three arguments [attr, old, new]
        # attr will be "data"
        # old will be the CDS dict before the trigger
        # new will be the CDS dict after the trigger
        assert attr == "data"
        # For some reason we can't get change events from the scale bar CDS,
        # but receive model_params CDS changes
        length = self.get_scalebar_length(self.scalebar_params.data)
        self.scalebar_input.value = length * self.params.scan_pixel_pitch * 1e9
        self.push()

    def update_cl(self, event):
        base = self.params
        params = base.adjust_camera_length(camera_length=event.new)
        self.update_with_force(self.model_params, self.params_update(params))

    def update_semiconv(self, event):
        params = self.params.derive(semiconv=event.new / 1000)
        self.update_with_force(self.model_params, self.params_update(params))

    def update_scale(self, event):
        base = self.params
        length_px = self.get_scalebar_length(self.scalebar_params.data)
        scan_pixel_pitch = event.new / 1e9 / length_px
        params = base.adjust_scan_pixel_pitch(scan_pixel_pitch)
        self.update_with_force(self.model_params, self.params_update(params))

    def move_scan_to(self, event):
        self.update_with_force(
            self.model_params,
            self.scan_pos_update(
                PixelYX(y=float(event.y), x=float(event.x)),
            ),
        )

    def move_rings_to(self, event):
        self.update_with_force(
            self.ring_params,
            self.detector_center_update(
                PixelYX(y=float(event.y), x=float(event.x)),
            ),
        )

    def descan_add_row(self):
        scan_pos = self.scan_pos
        center_pos = self.get_detector_center(self.ring_params.data)
        idx = (int(np.round(scan_pos.y)), int(np.round(scan_pos.x)))
        new_df = pd.DataFrame(
            [idx + (center_pos.y, center_pos.x)], columns=self.descan_columns
        )
        new_df = new_df.set_index(self.descan_index_cols)
        t = self.descan_fixpoint_table
        if idx in t.value.index:
            t.value.update(new_df)
            # Make sure the change is registered
            t.value = t.value
        else:
            t.value = pd.concat((t.value, new_df))

    def descan_delete_row(self, row):
        df = self.descan_fixpoint_table.value
        key = df.index[row]
        self.descan_fixpoint_table.value = df.drop(index=key)

    def descan_move_to_row(self, row):
        df = self.descan_fixpoint_table.value
        key = df.index[row]
        assert len(key) == 2
        scan_pos = PixelYX(y=key[0], x=key[1])
        self.update_with_force(self.model_params, self.scan_pos_update(scan_pos))

    def descan_drop(self):
        t = self.descan_fixpoint_table
        t.value = t.value.drop(t.value.index)

    def coord_add_row(self):
        scan_pos = self.scan_pos
        center_pos = self.get_detector_center(self.ring_params.data)
        idx = (int(np.round(scan_pos.y)), int(np.round(scan_pos.x)))
        new_df = pd.DataFrame(
            [idx + (center_pos.y, center_pos.x)], columns=self.coord_columns
        )
        new_df = new_df.set_index(self.coord_index_cols)
        t = self.coord_fixpoint_table
        if idx in t.value.index:
            t.value.update(new_df)
            # Make sure the change is registered
            t.value = t.value
        else:
            t.value = pd.concat((t.value, new_df))

    def coord_delete_row(self, row):
        df = self.coord_fixpoint_table.value
        key = df.index[row]
        self.coord_fixpoint_table.value = df.drop(index=key)

    def coord_move_to_row(self, row):
        df = self.coord_fixpoint_table.value
        key = df.index[row]
        detector = df.values[row]
        assert len(key) == 2
        scan_pos = PixelYX(y=key[0], x=key[1])
        self.update_with_force(self.model_params, self.scan_pos_update(scan_pos))
        detector_pos = PixelYX(y=detector[0], x=detector[1])
        self.update_with_force(self.ring_params, self.detector_center_update(detector_pos))

    def coord_drop(self):
        t = self.coord_fixpoint_table
        t.value = t.value.drop(t.value.index)

    def default_params(self) -> Parameters4DSTEM:
        ds = self.dataset
        if ds is None:
            scan_center = PixelYX(0.0, 0.0)
            detector_center = PixelYX(0.0, 0.0)
        else:
            scan_center = PixelYX(
                y=ds.shape.nav[0] / 2,
                x=ds.shape.nav[1] / 2,
            )
            detector_center = PixelYX(
                y=ds.shape.sig[0] / 2,
                x=ds.shape.sig[1] / 2,
            )
        return Parameters4DSTEM(
            overfocus=0.0,
            scan_pixel_pitch=1e-6,
            scan_center=scan_center,
            scan_rotation=0.0,
            camera_length=1.0,
            detector_pixel_pitch=50e-6,
            detector_center=detector_center,
            semiconv=1e-3,  # radian
            flip_factor=1.,
            # descan_error=DescanError(sxo_pxi=1, syo_pyi=-3)
        )

    def get_params(self, model_data):
        return self.deserialize(model_data["params"][0])

    @staticmethod
    def get_scan_pos(model_data):
        return PixelYX(y=model_data["scan_y"][0], x=model_data["scan_x"][0])

    @staticmethod
    def get_model_twothetas(model_data):
        # return np.array((0.01, 0.02, 0.03.))
        return model_data["twothetas"][0]

    @staticmethod
    def get_detector_center(ring_data):
        return PixelYX(
            y=ring_data["detector_cy"][0],
            x=ring_data["detector_cx"][0],
        )

    @staticmethod
    def get_scalebar_length(scalebar_data):
        start = np.array(
            (scalebar_data["scalebar_y"][0], scalebar_data["scalebar_x"][0])
        )
        stop = np.array(
            (scalebar_data["scalebar_y"][1], scalebar_data["scalebar_x"][1])
        )
        return np.linalg.norm(stop - start)

    @property
    def params(self) -> Parameters4DSTEM:
        return self.get_params(self.model_params.data)

    @property
    def scan_pos(self):
        return self.get_scan_pos(self.model_params.data)

    def get_preview(self):
        if self.nav_mode == "sumsig":
            nav_udf = SumSigUDF()
        elif self.nav_mode == "point":
            cy = self.params.detector_center.y
            cx = self.params.detector_center.x
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
        overfocus_udf = OverfocusUDF(overfocus_params={'params': self.params})

        res = self.ctx.run_udf(dataset=self.dataset, udf=(nav_udf, overfocus_udf))
        result = {}
        if self.nav_mode == "sumsig":
            result['nav'] = res[0]["intensity"].data
        elif self.nav_mode == "point":
            result['nav'] = res[0]["intensity"].data[..., 0]
        result['backprojected_sum'] = res[1]["backprojected_sum"].data
        result['corrected_point'] = res[1]["corrected_point"].data
        result['corrected_sum'] = res[1]["corrected_sum"].data
        return result

    def get_frames(self, pos: PixelYX):
        y = int(round(pos.y))
        x = int(round(pos.x))
        roi = self.dataset.roi[y, x]
        udfs = (
            PickUDF(),
            CorrectedPickUDF(overfocus_params={'params': self.params}),
        )
        res = self.ctx.run_udf(dataset=self.dataset, udf=udfs, roi=roi)
        return {
            'raw': res[0]['intensity'].raw_data[0],
            'corrected': res[1]['corrected'].raw_data[0],
            'backprojected': res[1]['backprojected'].raw_data[0],
        }

    @staticmethod
    def serialize(params: Parameters4DSTEM):
        return flax.serialization.to_state_dict(params.normalize_types())

    def deserialize(self, state_dict):
        res = flax.serialization.from_state_dict(
            target=self.start_params, state=state_dict
        ).normalize_types()
        return res

    def params_update(self, params: Parameters4DSTEM):
        return {"params": [self.serialize(params)]}

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

    def ring_update(self, model_data):
        params = self.get_params(model_data)
        ri, ro = ring_radii(
            params=params, twothetas=self.get_model_twothetas(model_data)
        )
        detector_center = get_center(
            params=params,
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

    @staticmethod
    def detector_center_update(detector_center: PixelYX):
        return {
            "detector_cy": [float(detector_center.y)],
            "detector_cx": [float(detector_center.x)],
        }

    def perform_descan_update(self):
        points = self.descan_fixpoint_table.value.reset_index().to_numpy()
        if len(points):
            new_params, residual = solve_tilt_descan_error_points(
                ref_params=self.params, points=jnp.array(points)
            )
            self.update_with_force(self.model_params, self.params_update(new_params))

    def _push(self):
        self.nav_fig.push(
            self.pick_fig, self.corr_point_fig, self.corr_pick_fig, self.corr_sum_fig,
            self.back_sum_fig, self.back_pick_fig,
        )

    def push(self):
        self._push()
        self.executor.submit(self._push)

    def update_with_force(self, cds: ColumnDataSource, update):
        cds.data.update(**update)
        self.push()
        holding = cds.document.callbacks._hold
        cds.document.callbacks.unhold()
        cds.data.update(**update)
        self.push()
        cds.document.callbacks.hold(holding)

    def center_correlation_regression(self):
        # Delicious! 🍝🍝🍝
        params = self.params
        ri, ro = ring_radii(params, self.get_model_twothetas(self.model_params.data))
        radius_outer = 1.2 * ro[0]
        if len(ri) >= 2:
            radius_outer = max(radius_outer, ri[1] / 2)
        # Make sure we stay away from other peaks
        pattern = BackgroundSubtraction(radius=ro[0], radius_outer=radius_outer)
        # Correction to same parameters, except descan error of 0
        ref_params = params.derive(descan_error=DescanError())
        mat = get_detector_correction_matrix(rec_params=params, ref_params=ref_params)
        # Scan positions in the dataset
        nav_shape = self.dataset.shape.nav
        y, x = np.mgrid[: nav_shape[0], : nav_shape[1]]
        # Calculate where the nominal center actually is on the detector
        expected_x = corrected_det_x(
            det_corr_x=params.detector_center.x,
            det_corr_y=params.detector_center.y,
            scan_x=x,
            scan_y=y,
            mat=mat,
        )
        expected_y = corrected_det_y(
            det_corr_x=params.detector_center.x,
            det_corr_y=params.detector_center.y,
            scan_x=x,
            scan_y=y,
            mat=mat,
        )
        # Subtract the expected position so that only the deviation remains
        dc = params.detector_center
        shifts = np.stack((expected_y - dc.y, expected_x - dc.x), axis=-1)
        aux_shifts = FastCorrelationUDF.aux_data(
            data=shifts,
            kind="nav",
            extra_shape=(2,),
            dtype=shifts.dtype,
        )
        res = run_fastcorrelation(
            ctx=self.ctx,
            dataset=self.dataset,
            peaks=np.array([params.detector_center]),
            match_pattern=pattern,
            zero_shift=aux_shifts,
            upsample=True,
        )

        # Following CoMUDF regression
        field = res["refineds"].data[:, :, 0, :]
        # Only keep deviation from expected value in regression
        field[..., 0] -= params.detector_center.y
        field[..., 1] -= params.detector_center.x

        inp = np.ones(field.shape[:-1] + (3,))
        y, x = np.ogrid[: field.shape[0], : field.shape[1]]
        inp[..., 1] = y
        inp[..., 2] = x
        reg_res = np.linalg.lstsq(
            inp.reshape((-1, 3)), field.reshape((-1, 2)), rcond=None
        )
        new_params, residual = solve_tilt_descan_error(
            ref_params=params, regression=reg_res[0]
        )
        self.model_params.data.update(**self.params_update(new_params))
        self.push()

    @property
    def layout(self):
        inputs = pn.layout.Row(self.scalebar_input, self.cl_input, self.semiconv_input,)
        raw_figs = pn.layout.Row(self.nav_fig.layout, self.pick_fig.layout)
        corrected_figs = pn.layout.Row(
            self.corr_point_fig.layout, self.corr_pick_fig.layout, self.corr_sum_fig.layout,
        )
        back_figs = pn.layout.Row(
            self.back_sum_fig.layout, self.back_pick_fig.layout
        )
        descan_buttons = pn.layout.Row(
            self.record_button,
            self.apply_button,
            self.clear_button,
            self.correlate_button,
        )
        self.descan_label = pn.pane.Markdown("# Descan correction", )
        descan_section = pn.layout.Column(
            self.descan_label,
            self.descan_fixpoint_table,
            descan_buttons
        )
        coord_buttons = pn.layout.Row(
            self.coord_record_button,
            self.coord_apply_button,
            self.coord_clear_button,
            self.optimize_button,
        )
        self.coord_label = pn.pane.Markdown("# Coordinate system calibration", )
        coord_section = pn.layout.Column(
            self.coord_label,
            self.coord_fixpoint_table,
            coord_buttons
        )
        table_row = pn.layout.Row(descan_section, coord_section)
        return pn.layout.Column(
            inputs,
            raw_figs,
            table_row,
            corrected_figs,
            back_figs,
        )
