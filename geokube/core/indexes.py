from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Self

import dask.array as da
import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from . import axis
from .indexers import get_indexer


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
        method: str | None = None,
        tolerance: npt.ArrayLike | None = None
    ) -> xr.core.indexing.IndexSelResult:
        if len(labels) != 1:
            raise ValueError("'labels' can contain exactly one item")

        coord_axis, label = next(iter(labels.items()))
        # TODO: Consider if this is necessary.
        if coord_axis != self.coord_axis:
            raise ValueError("'labels' contain a wrong axis")

        # idx = _get_indexer_1d(self.data, label, method, tolerance)[0]
        data = self.data
        idx = get_indexer(
            [self.data.magnitude],
            [get_magnitude(label, data.units)],
            method=method,
            tolerance=tolerance,
            return_all=False
        )

        print(idx)

        return xr.core.indexing.IndexSelResult({self.dims[0]: idx[0]})


@dataclass(frozen=True, slots=True)
class TwoDimHorPointsIndex(xr.core.indexes.Index):
    latitude: pint.Quantity
    longitude: pint.Quantity
    dims: tuple[Hashable]

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any]
    ) -> Self:
        if len(variables) != 2:
            raise ValueError("'variables' can contain exactly two items")

        try:
            lat = variables[axis.latitude]
            lon = variables[axis.longitude]
        except KeyError as err:
            raise ValueError(
                "'variables' must contain both latitude and longitude"
            ) from err

        lat_data, lon_data = lat.data, lon.data
        if not (
            isinstance(lat_data, pint.Quantity)
            and isinstance(lon_data, pint.Quantity)
        ):
            raise TypeError("'variables' must contain data of type 'Quantity'")

        dims = lat.dims
        if lon.dims != dims:
            raise ValueError("'variables' must have the same dimensions")
        if len(dims) != 1:
            raise ValueError("'variables' must contain one-dimensional data")

        return cls(latitude=lat_data, longitude=lon_data, dims=dims)

    def sel(
        self,
        labels: dict[Hashable, slice | npt.ArrayLike | pint.Quantity],
        method: str | None = None,
        tolerance: npt.ArrayLike | None = None
    ) -> xr.core.indexing.IndexSelResult:
        # TODO: Try this approach with 2-D and N-D data.
        if len(labels) != 2:
            raise ValueError("'labels' can contain exactly two items")

        try:
            lat = labels[axis.latitude]
            lon = labels[axis.longitude]
        except KeyError as err:
            raise ValueError(
                "'labels' must contain both latitude and longitude"
            ) from err

        # idx = _get_indexer_nd(self.latitude, self.longitude, lat, lon)[0]
        lat_, lon_ = self.latitude, self.longitude
        idx = get_indexer(
            [lat_.magnitude, lon_.magnitude],
            [get_magnitude(lat, lat_.units), get_magnitude(lon, lon_.units)],
            combine_result=True,
            method=method,
            tolerance=np.inf if tolerance is None else tolerance,
            return_all=False
        )

        print(idx)

        return xr.core.indexing.IndexSelResult({self.dims[0]: idx[0]})


def _get_array_like_magnitude(
    data: npt.ArrayLike | da.Array | pint.Quantity, units: pint.Unit
) -> npt.ArrayLike | da.Array:
    if isinstance(data, pint.Quantity):
        if data.units != units:
            data = data.to(units)
        return data.magnitude
    return data


def _get_slice_magnitude(data: slice, units: pint.Unit) -> slice:
    if data.step is not None:
        raise ValueError("'data' must have the step 'None'")
    start = _get_array_like_magnitude(data.start, units)
    stop = _get_array_like_magnitude(data.stop, units)
    return slice(start, stop)


def get_magnitude(
    data: slice | npt.ArrayLike | da.Array | pint.Quantity, units: pint.Unit
) -> slice | npt.ArrayLike | da.Array:
    match data:
        case slice():
            return _get_slice_magnitude(data, units)
        case _:
            return _get_array_like_magnitude(data, units)


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
        # TODO: Consider using `_get_indexer_nd` here.
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
