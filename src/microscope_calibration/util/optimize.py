import numpy as np
from scipy. optimize import shgo
from skimage.measure import blur_effect
from typing import TYPE_CHECKING, Iterable, Callable, Optional

from microscope_calibration.common.stem_overfocus import OverfocusParams

if TYPE_CHECKING:
    from libertem.udf.base import UDF
    from libertem.api import Context
    from libertem.io.dataset.base import DataSet

    from microscope_calibration.udf.stem_overfocus import OverfocusUDF


def make_overfocus_loss_function(
        params: OverfocusParams,
        ctx: 'Context',
        dataset: 'DataSet',
        overfocus_udf: 'OverfocusUDF',
        blur_function: Optional[Callable] = None,
        extra_udfs: Iterable['UDF'] = (),
        callback: Optional[Callable] = None,
        **kwargs):
    '''
    Build a parameter mapping and loss function to optimize

    This maps the :code:`scan_rotation` and :code:`overfocus` parameters
    for :class:`OverfocusUDF` in such a way that the starting value is at
    0 and a sensible interval for optimization is :code`[-10, 10]` based
    on the starting parameters passed as an argument.

    It also constructs a loss function that updates the parameters of
    the :class:`OverfocusUDF` passed as parameter :code:`overfocus_udf`,
    runs it with the specified context, dataset and extra parameters,
    and then returns the blur effect of the image.

    Parameters
    ----------
    params
        Starting value for optimization
    ctx
        Run UDF using this Context
    dataset
        Run UDF with this dataset
    overfocus_udf
        Instance of :class:`OverfocusUDF` to use. Updating and using an existing instance
        allows using live plots.
    blur_function
        Function to calculate the blur of the :code:`shifted_sum` result buffer.
        By default, :code:`skimage.measure.blur_effect` is used.
    extra_udfs
        Iterable of other UDFs to run alongside
    callback
        Function that is called for each loop with the loss function
        arguments, current parameters, UDF results and current loss
    **kwargs
        Extra arguments to pass to the :func:`~libertem.api.Context.run_udf` call,
        such as plots.

    Returns
    -------
    make_new_params
        Function that updates a copy of the passed input parameters
        with an argument vector of the loss function. Since this vector
        is mapped into a [-10, 10] interval, this function is created to
        invert the mapping. This can be used
        to create parameters from the minimization result.
    loss
        Function that can be called with :func:`scipy.optimize.minimize`. Starting value is
        [0, 0], sensible range is ([-10, 10], [-10, 10]).
    '''
    # Rotate and scale the angle so that the optimizer works between +-10,
    # corresponding to +- 5 deg
    rotation_diff = params['scan_rotation']
    rotation_scale = 1
    # Values to shift and scale the overfocus so that the optimizer works between +-10
    overfocus_diff = params['overfocus']
    overfocus_scale = 40 / np.abs(params['overfocus'])

    if blur_function is None:
        blur_function = blur_effect

    def make_new_params(args) -> OverfocusParams:
        '''
        Map parameters from +-10 to original range

        Parameters
        ----------

        args
            (scan_rotation, overfocused) mapped to +- 10
        '''
        transformed_rotation, transformed_overfocus = args
        rotation = transformed_rotation / rotation_scale + rotation_diff
        overfocus = transformed_overfocus / overfocus_scale + overfocus_diff
        param_copy = params.copy()
        param_copy['overfocus'] = overfocus
        param_copy['scan_rotation'] = rotation
        return param_copy

    def loss(args) -> float:
        '''
        Loss function for optimizer to call

        This calls the UDF with the appropriate parameters
        and returns the blur as a metric. It can perform additional actions
        per iteration through the :code:`callback`, :code:`extra_udfs` and
        :code:`kwargs` parameters of the surrounding function.

        Parameters
        ----------

        args
            (scan_rotation, overfocused) mapped to +-10 range
        '''
        param_copy = make_new_params(args)
        overfocus_udf.params.overfocus_params.update(param_copy)
        res = ctx.run_udf(dataset=dataset, udf=[overfocus_udf] + list(extra_udfs), **kwargs)
        blur = blur_function(res[0]['shifted_sum'].data)
        if callback is not None:
            callback(args, overfocus_udf.params.overfocus_params, res, blur)
        return blur

    return make_new_params, loss


def optimize(loss, bounds=None, minimizer_kwargs=None, **kwargs):
    '''
    Convenience function to call :func:`scipy.optimize.shgo`

    This calls :func:`scipy.optimize.shgo` with sensible bounds and minimizer
    method for a loss function created with
    :func:`make_overfocus_loss_function`. Additional kwargs are passed to
    :func:`scipy.optimize.shgo`.
    '''
    if bounds is None:
        bounds = [(-10, 10), (-10, 10)]

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'COBYLA'}
    res = shgo(
        func=loss,
        bounds=bounds,
        minimizer_kwargs=minimizer_kwargs,
        **kwargs
    )
    return res
