from collections.abc import Sequence
from enum import Enum, unique
from typing import TypeVar

import numpy as np
import numpy.typing as npt


# TODO: Check the type hints, especially for the return types.

# NOTE: The function `scipy.spatial.distance.cdist` does not support the types:
# `datetime64`, `timedelta64`, `str_`, and `object`. Documentation for this
# function:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

# NOTE: The function `numpy.allclose` does not support the types: `datetime64`,
# `timedelta64`, `str_`, and `object`. Documentation for this function:
# https://numpy.org/doc/stable/reference/generated/numpy.allclose.html


@unique
class Metric(Enum):
    CITY_BLOCK = 'city_block'
    EUCLIDEAN = 'euclidean'


@unique
class Accuracy(Enum):
    EXACT = 'exact'
    NEAREST = 'nearest'


@unique
class ReturnType(Enum):
    BOOL = 'bool'
    INT = 'int'
    DEFAULT = 'default'


_IndexArrayT = TypeVar(
    '_IndexArrayT', npt.NDArray[np.bool_], npt.NDArray[np.int_]
)


def get_slice_indexer(
    x_data: Sequence[np.ndarray],
    y_data: Sequence[slice],
    combine_result: bool = False,
    return_type: str | ReturnType = ReturnType.BOOL
) -> tuple[_IndexArrayT] | list[_IndexArrayT] | list[tuple[_IndexArrayT]]:
    # Checking if the lengths of `x_data` and `y_data` match. -----------------
    if len(x_data) != len(y_data):
        raise ValueError("'x_data' and 'y_data' must have the same lengths")

    # Analyzing `x_data`. -----------------------------------------------------
    x_shapes = {x_arr.shape for x_arr in x_data}
    if len(x_shapes) != 1:
        raise ValueError("'x_data' must have arrays of equal shapes")
    x_shape = x_shapes.pop()

    # Creating masks. ---------------------------------------------------------
    masks = []
    true_mask = np.full(shape=x_shape, fill_value=True, dtype=np.bool_)
    for x_arr, y_slice in zip(x_data, y_data):
        if y_slice.step is not None:
            raise ValueError("'y_data' must have the step 'None'")
        # NOTE: It is important to use `np.dtype.kind` in `np.ndarray.astype`
        # to properly handle date and time units/resolutions.
        x_dtype = x_arr.dtype
        x_dtype_ = x_dtype.kind
        y_start, y_stop = y_slice.start, y_slice.stop
        if y_start is None:
            if y_stop is None:
                mask = true_mask
            else:
                y_stop_ = np.asarray(y_slice.stop)
                if (
                    not np.issubdtype(x_dtype, np.str_)
                    and np.issubdtype(y_stop_.dtype, np.str_)
                ):
                    y_stop_ = y_stop_.astype(x_dtype_)
                mask = x_arr <= y_stop_
        else:
            if y_stop is None:
                y_start_ = np.asarray(y_slice.start)
                if (
                    not np.issubdtype(x_dtype, np.str_)
                    and np.issubdtype(y_start_.dtype, np.str_)
                ):
                    y_start_ = y_start_.astype(x_dtype_)
                mask = x_arr >= y_start_
            else:
                y_start_ = np.asarray(y_slice.start)
                y_stop_ = np.asarray(y_slice.stop)
                if not np.issubdtype(x_dtype, np.str_):
                    if np.issubdtype(y_start_.dtype, np.str_):
                        y_start_ = y_start_.astype(x_dtype_)
                    if np.issubdtype(y_stop_.dtype, np.str_):
                        y_stop_ = y_stop_.astype(x_dtype_)
                lower_bound, upper_bound = sorted([y_start_, y_stop_])
                mask = (x_arr >= lower_bound) & (x_arr <= upper_bound)
        masks.append(mask)

    # Calculating indices. ----------------------------------------------------
    if combine_result:
        idx = true_mask
        for mask in masks:
            idx &= mask
        if ReturnType(return_type) is ReturnType.INT:
            return np.nonzero(idx)
        return (idx,)

    if ReturnType(return_type) is ReturnType.INT:
        return [np.nonzero(mask) for mask in masks]
    return masks


def get_array_indexer(
    x_data: Sequence[np.ndarray],
    y_data: Sequence[npt.ArrayLike],
    metric: str | Metric | None = None,
    method: str | Accuracy | None = None,
    tolerance: npt.ArrayLike | None = None,
    return_all: bool = True,
    return_type: str | ReturnType = ReturnType.INT
) -> tuple[_IndexArrayT] | list[_IndexArrayT] | list[tuple[_IndexArrayT]]:
    # Checking if the lengths of `x_data` and `y_data` match. -----------------
    if len(x_data) != len(y_data):
        raise ValueError("'x_data' and 'y_data' must have the same lengths")

    # Analyzing `x_data`. -----------------------------------------------------
    x_shapes = {x_arr.shape for x_arr in x_data}
    if len(x_shapes) != 1:
        raise ValueError("'x_data' must have arrays of equal shapes")
    x_shape = x_shapes.pop()
    # x_shape = x_data[0].shape
    x_n_dims = x_data[0].ndim
    x_dtype = np.result_type(*x_data)
    # NOTE: It is important to use `np.dtype.kind` in `np.ndarray.astype` to
    # properly handle date and time units/resolutions.
    x_dtype_ = x_dtype.kind

    # Analyzing `y_data` and converting to NumPy arrays. ----------------------
    y_data_: list[np.ndarray] = []
    y_shapes: set[tuple[int, ...]] = set()
    for y_arr in y_data:
        # NOTE: The case where `x_data` has integers and `y_data` has floats,
        # seems particularly dangerous.
        y_arr_ = np.array(y_arr, copy=False, ndmin=1)
        if (
            not np.issubdtype(x_dtype, np.str_)
            and np.issubdtype(y_arr_.dtype, np.str_)
        ):
            y_arr_ = y_arr_.astype(x_dtype_)
        y_data_.append(y_arr_)
        y_shapes.add(y_arr_.shape)
    # NOTE: A similar result can be obtained by using `numpy.broadcast`
    # iteratively  in the previous `for` loop.
    if len(y_shapes) != 1:
        y_data_ = np.broadcast_arrays(*y_data_)
    y_shape = y_data_[0].shape
    y_size = y_shape[0]
    y_n_dims = len(y_shape)

    # Reshaping `x_data` and `y_data`. ----------------------------------------
    new_axis = (np.newaxis,)  # Actually, `np.newaxis` is just `None`.
    old_axes = (...,)
    x_reshape_axes = new_axis * y_n_dims + old_axes
    x_data_ = [arr[x_reshape_axes] for arr in x_data]
    y_reshape_axes = old_axes + new_axis * x_n_dims
    y_data_ = [arr[y_reshape_axes] for arr in y_data_]

    # Calculating the differences between `x_data` and `y_data`. --------------
    # NOTE: For Boolean, object, byte-string, and string arrays, the default
    # metric is city block. For time delta, the default metric is also city
    # block, because squares are not supported. For other data types, it is
    # Euclidean.
    # NOTE: For Boolean, object, byte-string, and string arrays, the default
    # method is exact. For other data types, it is nearest.
    # NOTE: For Boolean, object, byte-string, and string arrays, the tolerance
    # is 0. The value provided as a function argument is ignored.
    if x_dtype_ in {'b', 'O', 'S', 'U'}:
        # NOTE: Equal values in any dimension correspond to the difference 0,
        # while different values yield the difference 1.
        diffs = [
            (x_arr != y_arr).astype(np.int32)
            for x_arr, y_arr in zip(x_data_, y_data_)
        ]
        if metric is None:
            metric = Metric.CITY_BLOCK
        if method is None:
            method = Accuracy.EXACT
    else:
        diffs = [x_arr - y_arr for x_arr, y_arr in zip(x_data_, y_data_)]
        if diffs[0].dtype.kind == 'm':
            if metric is None:
                metric = Metric.CITY_BLOCK
            if method is None:
                method = Accuracy.EXACT
        else:
            if metric is None:
                metric = Metric.EUCLIDEAN
            if method is None:
                method = Accuracy.NEAREST
            # TODO: Consider the default value of `tolerance`.
            # if tolerance is None and return_all:
            #     tolerance = 1e-08
            if tolerance is None:
                # tolerance = 1e-08 if return_all else np.inf
                tolerance = 1e-08
    # NOTE: For time delta, the default metric is city block, because squares
    # are not supported. For other data types, it is Euclidean.
    match metric := Metric(metric):
        case Metric.CITY_BLOCK:
            tot_diff = sum(np.abs(diff) for diff in diffs)
        case Metric.EUCLIDEAN:
            tot_diff = np.sqrt(sum(diff * diff for diff in diffs))
        case _:
            raise ValueError(f"'metric' with the name {metric} is not allowed")
    if tolerance is None:
        # NOTE: In this case, `tolerance` is ignored.
        allow_diff = np.full_like(tot_diff, fill_value=True, dtype=np.bool_)
    else:
        tol = np.asarray(
            tolerance if Accuracy(method) is Accuracy.NEAREST else 0,
            dtype=tot_diff.dtype
        )
        allow_diff = tot_diff <= tol

    # Calculating indices. ----------------------------------------------------
    # TODO: Modify the rest of the function, so that it supports
    # multi-dimensional `y_data`.
    if y_n_dims > 1:
        raise NotImplementedError(
            "'y_data' is multi-dimensional, which is currently not supported"
        )

    if return_all:
        idx = [allow_diff[i] for i in range(y_size)]
        if ReturnType(return_type) is ReturnType.INT:
            return [np.nonzero(mask) for mask in idx]
        return idx

    if ReturnType(return_type) is ReturnType.BOOL:
        raise NotImplementedError("'return_type' cannot be 'bool'")

    # Selecting the indices that correspond to the minimal differences between
    # `x_data` and `y_data`.
    # TODO: Improve vectorization.
    # TODO: Consider replacing `numpy.unravel_index` with
    # `numpy.argwhere`, using the constructs like:
    # `np.argwhere(tot_diff[i] == tot_diff[i].min())[0]`.
    # err_code = (-1,) * x_n_dims
    idx_nd = []
    for i in range(y_size):
        if allow_diff[i].any():
            idx_nd.append(
                np.unravel_index(indices=tot_diff[i].argmin(), shape=x_shape)
            )
        else:
            # idx_nd.append(err_code)
            raise ValueError("'y_data' contains a value that cannot be found")
    # idx_nd = tuple(
    #     np.unravel_index(indices=sum_sq_diff[i].argmin(), shape=shape)
    #     for i in range(n_vals)
    # )
    idx = np.array(idx_nd, dtype=np.int64)
    return tuple(idx[:, i] for i in range(x_n_dims))


def get_indexer(
    x_data: Sequence[np.ndarray],
    y_data: Sequence[slice | npt.ArrayLike],
    combine_result: bool = False,
    metric: str | Metric | None = None,
    method: str | Accuracy | None = None,
    tolerance: npt.ArrayLike | None = None,
    return_all: bool = True,
    return_type: str | ReturnType = ReturnType.DEFAULT
) -> tuple[_IndexArrayT] | list[_IndexArrayT] | list[tuple[_IndexArrayT]]:
    match list(y_data):
        case [slice(), *_]:
            return get_slice_indexer(
                x_data, y_data, combine_result, return_type
            )
        case _:
            return get_array_indexer(
                x_data, y_data, metric, method, tolerance, return_all,
                return_type
            )
