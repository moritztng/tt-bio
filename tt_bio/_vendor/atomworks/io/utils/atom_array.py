from collections.abc import Callable

import biotite.structure as struc
import numpy as np

NUMPY_REDUCE_FUNCS = {
    np.any: np.logical_or.reduceat,
    np.all: np.logical_and.reduceat,
    np.maximum: np.maximum.reduceat,
    np.minimum: np.minimum.reduceat,
    np.max: np.maximum.reduceat,
    np.min: np.minimum.reduceat,
    np.add: np.add.reduceat,
    np.sum: np.add.reduceat,
}


def apply_and_spread(
    segment_start_stop_idxs: np.ndarray, data: np.ndarray, function: Callable, axis: int | None = None
) -> np.ndarray:
    """
    Apply a function segment-wise and then spread the result to the original data size.

    This function applies a given function to segments of the input data and then
    spreads the result back to the original data size, effectively assigning the
    segment-wise result to all elements within each segment.

    Args:
        segment_start_stop_idxs: A 1D array indicating the start and stop indices
            of each segment.  This is expected to be in the format returned by
            `biotite.structure.segments.get_segment_starts`.
        data: The input data array.
        function: The function to apply to each segment.  This function should
            take a segment of the data array as input and return a single value
            or an array of reduced values.
        axis: The axis along which to apply the function. If `None`, the function
            is applied to the entire segment.

    Returns:
        A new array with the same shape as `data`, where the result of the
        function applied to each segment has been spread across the elements
        of that segment.

    Example:
        >>> import numpy as np
        >>> segment_start_stop_idxs = np.array([0, 3, 6])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> result = apply_and_spread(segment_start_stop_idxs, data, np.sum)
        >>> print(result)
        [ 6  6  6 15 15 15]
    """
    if function in NUMPY_REDUCE_FUNCS and axis is None:
        data_after_apply = NUMPY_REDUCE_FUNCS[function](data, segment_start_stop_idxs[:-1])
    else:
        data_after_apply = struc.segments.apply_segment_wise(segment_start_stop_idxs, data, function, axis)
    data_after_spread = struc.segments.spread_segment_wise(segment_start_stop_idxs, data_after_apply)
    return data_after_spread
