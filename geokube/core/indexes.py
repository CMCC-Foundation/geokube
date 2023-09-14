from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Self

import dask.array as da
import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from . import axis


# TODO: Consider making this module and `TwoDimIndex` internal.


@dataclass(frozen=True, slots=True)
class OneDimIndex(xr.core.indexes.Index):
    data: pint.Quantity
    dims: tuple[Hashable]
    coord_axis: axis.Axis

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any]
    ) -> Self:
        if len(variables) != 1:
            raise ValueError("'variables' can contain exactly one item")

        coord_axis, var = next(iter(variables.items()))
        if not isinstance(coord_axis, axis.Axis):
            raise TypeError("'variables' key must be an instance of 'Axis'")

        data, dims = var.data, var.dims
        if not isinstance(data, pint.Quantity):
            raise TypeError("'variables' must contain data of type 'Quantity'")
        if len(dims) != 1:
            raise ValueError("'variables' value must be be one-dimensional")

        return cls(data=data, dims=dims, coord_axis=coord_axis)

    def sel(
        self,
        labels: dict[Hashable, slice | npt.ArrayLike | pint.Quantity],
        method: str | None = "nearest",
        tolerance: int | float | None = None
    ) -> xr.core.indexing.IndexSelResult:
        if len(labels) != 1:
            raise ValueError("'labels' can contain exactly one item")

        coord_axis, label = next(iter(labels.items()))
        # TODO: Consider if this is necessary.
        if coord_axis != self.coord_axis:
            raise ValueError("'labels' contain a wrong axis")

        idx = _get_indexer(self.data, label, method, tolerance)[0]

        return xr.core.indexing.IndexSelResult({self.dims[0]: idx})


def _get_magnitude(
    data: npt.ArrayLike | da.Array | pint.Quantity, units: pint.Unit
) -> npt.ArrayLike | da.Array:
    if isinstance(data, pint.Quantity):
        if data.units != units:
            data = data.to(units)
        return data.magnitude
    return data


def _get_indexer(
    quantity: pint.Quantity,
    label: slice | npt.ArrayLike | pint.Quantity,
    method: str | None = 'nearest',
    tolerance: int | float | None = None,
) -> npt.NDArray[np.intp] | tuple[npt.NDArray[np.intp], ...]:
    data, units = quantity.magnitude, quantity.units
    dtype = data.dtype

    match label:
        case slice():
            start = _get_magnitude(label.start, units)
            stop = _get_magnitude(label.stop, units)
            if label.step is not None:
                raise ValueError("'label' must have step 'None'")
            lb_, ub_ = sorted([start, stop])
            # lb_, ub_ = (start, stop) if start < stop else (stop, start)
            lb_arr = np.asarray(lb_, dtype=dtype)
            ub_arr = np.asarray(ub_, dtype=dtype)
            return np.nonzero((data >= lb_arr) & (data <= ub_arr))
        case _:
            arr_lib = da if isinstance(data, da.Array) else np
            n_dims, shape = data.ndim, data.shape
            data_ = data[np.newaxis, ...]
            label_vals = _get_magnitude(label, units)
            vals = np.asarray(label_vals, dtype=dtype).reshape(-1)
            vals_ = vals[(np.s_[:],) + (np.newaxis,) * n_dims]
            n_vals = vals.size

            if dtype.kind in {'O', 'S', 'U'}:
                method = None
                abs_diff = (data_ != vals_).astype(np.int64)
            else:
                abs_diff = arr_lib.abs(data_ - vals_)

            idx = tuple(
                np.empty(shape=n_vals, dtype=np.int64) for _ in range(n_dims)
            )
            for i in range(n_vals):
                raw_idx = np.unravel_index(
                    indices=abs_diff[i].argmin(), shape=shape
                )
                for j, idx_item in enumerate(idx):
                    idx_item[i] = raw_idx[j]

            match method:
                case 'nearest':
                    if dtype.kind in {'m', 'M'}:
                        # NOTE: Tolerance does not apply for `datetime64` and
                        # `timedelta64`.
                        acc_check = True
                    else:
                        tol = float(
                            'inf'
                            if tolerance is None else
                            _get_magnitude(tolerance, units)
                        )
                        acc_check = arr_lib.allclose(
                            data[idx], vals, rtol=0, atol=tol
                        )
                case None:
                    acc_check = bool((data[idx] == vals).all())
                case _:
                    raise ValueError(f"'method' cannot have be {method}")

            if not acc_check:
                raise ValueError("'values' contain items that cannot be found")

            return idx


@dataclass(frozen=True, slots=True)
class TwoDimIndex(xr.core.indexes.Index):
    variables: dict[str, xr.Variable]
    dims: tuple[Hashable, ...]

    # TODO: Consider using `Axis` instead of `str` for dims.

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> Self:
        variables_ = dict(variables)

        if len(variables_) != 2:
            raise ValueError("'variables' must be a mapping with two items")

        vars_ = iter(variables.values())
        dims = next(vars_).dims
        if next(vars_).dims != dims:
            raise ValueError("'variables' must have same dimensions")
        if len(dims) != 2:
            raise ValueError("'variables' must be two-dimensional")

        return cls(variables_, dims)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | xr.Variable]
    ) -> Self | None:
        # TODO: Consider if this method is going to be implemented.
        pass

    def sel(self, labels: dict[Any, Any]) -> xr.core.indexing.IndexSelResult:
        dims = self.dims
        vars_ = self.variables
        vars_iter = iter(vars_)
        x_name = next(vars_iter)
        x_vals = vars_[x_name].to_numpy()
        y_name = next(vars_iter)
        y_vals = vars_[y_name].to_numpy()

        indices = {}

        for name, var in vars_.items():
            if (idx := labels.get(name)) is not None:
                if isinstance(idx, slice):
                    # TODO: Consider the case when `idx` has `None` for `start`
                    # or `stop`.
                    slices = True
                    lower_bound, upper_bound = sorted([idx.start, idx.stop])
                    vals = var.to_numpy()
                    indices[name] = (
                        (vals >= lower_bound) & (vals <= upper_bound)
                    )
                else:
                    indices[name] = np.array(idx, ndmin=1)
                    slices = False

        x_idx = indices[x_name]
        y_idx = indices[y_name]
        if slices:
            nonzero_idx = np.nonzero(x_idx & y_idx)
            sel_idx = {
                dim: slice(incl_idx.min(), incl_idx.max() + 1)
                for dim, incl_idx in zip(self.dims, nonzero_idx)
            }
        else:
            # Calculating the squares of the Euclidean distance.
            x_data = x_vals[np.newaxis, :, :]
            y_data = y_vals[np.newaxis, :, :]
            x_pts = x_idx[:, np.newaxis, np.newaxis]
            y_pts = y_idx[:, np.newaxis, np.newaxis]
            # x_pts = x_idx.reshape(-1, 1, 1)
            # y_pts = y_idx.reshape(-1, 1, 1)
            x_diff = x_data - x_pts
            y_diff = y_data - y_pts
            sum_sq_diff = x_diff * x_diff + y_diff * y_diff

            # Selecting the indices that correspond to the squares of the
            # Euclidean distance.
            # TODO: Improve vectorization.
            # TODO: Consider replacing `numpy.unravel_index` with
            # `numpy.argwhere`, using the constructs like:
            # `np.argwhere(diff_sq[i] == diff_sq[i].min())[0]`.
            n_dims, *shape = sum_sq_diff.shape
            idx_nd = tuple(
                np.unravel_index(indices=sum_sq_diff[i].argmin(), shape=shape)
                for i in range(n_dims)
            )
            idx_ = np.array(idx_nd, dtype=np.int64)
            sel_idx = {dims[0]: idx_[:, 0], dims[1]: idx_[:, 1]}

        return xr.core.indexing.IndexSelResult(sel_idx)
