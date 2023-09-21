from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Self

import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from . import axis
from .indexers import get_indexer
from .quantity import get_magnitude


# TODO: Consider making this module and `TwoDimIndex` internal.
# TODO: Consider removing redundant checks in `.from_variables`.


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

        return xr.core.indexing.IndexSelResult({self.dims[0]: idx[0]})


@dataclass(frozen=True, slots=True)
class TwoDimVertProfileIndex(xr.core.indexes.Index):
    vertical: pint.Quantity
    data: pint.Quantity
    name: str
    dims: tuple[Hashable, Hashable]

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any]
    ) -> Self:
        if len(variables) != 1:
            raise ValueError("'variables' can contain exactly one item")

        try:
            vert = variables[axis.vertical]
        except KeyError as err:
            raise ValueError(
                "'variables' must contain the vertical coordinate"
            ) from err

        try:
            opts = dict(options)
            data = opts.pop('data')
            name = str(opts.pop('name'))
        except KeyError as err:
            raise ValueError(
                "'options' must contain the data and name of the field"
            ) from err

        if opts:
            raise ValueError(
                "'options' cannot contain anything but the data and name of "
                "the field"
            )

        vert_data = vert.data
        data_data = data.data
        if not (
            isinstance(vert_data, pint.Quantity)
            and isinstance(data_data, pint.Quantity)
        ):
            raise TypeError(
                "'variables' and 'options' must contain data of type "
                "'Quantity'"
            )

        dims = vert.dims
        if data.dims != dims:
            raise ValueError(
                "'variables' and 'options' must have data with the same "
                "dimensions"
            )
        if set(dims) != {'_profiles', '_levels'}:
            raise ValueError(
                "'variables' and 'options' must contain data with the "
                "dimensions '_profiles' and '_levels'"
            )

        return cls(vertical=vert_data, data=data_data, name=name, dims=dims)

    def sel(
        self,
        labels: dict[Hashable, slice | npt.ArrayLike | pint.Quantity],
        method: str | None = None,
        tolerance: npt.ArrayLike | None = None
    ) -> xr.core.indexing.IndexSelResult:
        if len(labels) != 1:
            raise ValueError("'labels' can contain exactly one item")

        try:
            vert_labels = labels[axis.vertical]
        except KeyError as err:
            raise ValueError(
                "'labels' must contain the vertical coordinate"
            ) from err

        vert = self.vertical
        vert_mag, vert_units = vert.magnitude, vert.units
        data = self.data
        dims = self.dims

        match vert_labels:
            case slice():
                mask_: list[npt.NDArray[np.bool_]] = get_indexer(
                    [vert_mag], [get_magnitude(vert_labels, vert_units)]
                )
                mask: npt.NDArray[np.bool_] = mask_[0]
                # Finding the index slice that removes redundant columns from
                # the begining and the end.
                keep_cols_mask = mask.any(axis=0)
                keep_cols_idx = keep_cols_mask.nonzero()[0]
                # keep_cols_slice = slice(
                #     keep_cols_idx[0], keep_cols_idx[-1] + 1
                # )
                keep_cols_slice = slice(
                    keep_cols_idx.min(), keep_cols_idx.max() + 1
                )
                # Finding the index slice that removes redundant rows from the
                # begining and the end.
                keep_rows_mask = mask.any(axis=1)
                keep_rows_idx = np.nonzero(keep_rows_mask)[0]
                # keep_rows_slice = slice(
                #     keep_rows_idx[0], keep_rows_idx[-1] + 1
                # )
                keep_rows_slice = slice(
                    keep_rows_idx.min(), keep_rows_idx.max() + 1
                )
                idx = (keep_rows_slice, keep_cols_slice)

                return xr.core.indexing.IndexSelResult(
                    dim_indexers=dict(zip(self.dims, idx))
                )

                # NOTE: It seems that this approach is not working.
                # new_mask = mask[idx]
                # new_mask_ = xr.Variable(dims=dims, data=new_mask)
                # new_vert = np.where(new_mask, vert_mag[idx], np.nan)
                # new_vert_ = xr.Variable(
                #     dims=dims, data=pint.Quantity(new_vert, vert_units)
                # )
                # new_data = np.where(new_mask, data.magnitude[idx], np.nan)
                # new_data_ = xr.Variable(
                #     dims=dims, data=pint.Quantity(new_data, data.units)
                # )
                # return xr.core.indexing.IndexSelResult(
                #     dim_indexers=dict(zip(self.dims, idx)),
                #     variables={
                #         'mask': new_mask_,
                #         axis.vertical: new_vert_,
                #         self.name: new_data_
                #     }
                # )
            case _:
                raise NotImplementedError()


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
