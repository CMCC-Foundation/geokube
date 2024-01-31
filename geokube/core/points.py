# TODO: Consider if this file is redundant.

from enum import Enum, unique

import numpy as np
import pint
import pandas as pd
import xarray as xr

from . import axis


@unique
class CopyApproach(Enum):
    # Never allow copying data. Raise error if necessary.
    NEVER = "never"
    # Allow copying data only if it is necessary to perform an operation.
    REQUIREMENT = "requirement"
    # Allow copying data only if it is necessary to get a contiguous result.
    CONTIGUOUS = "contiguous"
    # Always force copying data.
    ALWAYS = "always"


def to_points_dict(
    name: str,
    dset: xr.Dataset,
    allow_copy: str | CopyApproach = "requirement",
    omit_data: bool = False,
) -> dict[axis.Axis, pint.Quantity]:
    darr = dset[name]
    dims = dset.dims
    dims_ = set(dims)
    coords_copy = dict(dset.coords)
    # dim_axis_map = {
    #     dim: axis_ for axis_ in coords_copy if (dim := f'_{axis_}') in dims_
    # }
    dim_axis_map = {
        dim: axis_ for axis_ in coords_copy if (dim := f"{axis_}") in dims_
    }

    n_vals = n_reps = darr.size
    # n_vals = 1
    # for dim_axis in dim_axes:
    #     n_vals *= coords_copy[dim_axis].size
    # n_reps = n_vals
    # coords_copy = coords.copy()
    idx, data = {}, {}

    # Dimension coordinates.
    for dim in dims:
        dim_axis = dim_axis_map.get(dim, dim[1:])
        coord = coords_copy.pop(dim_axis, dset[dim])
        n_coord_vals = coord.size
        n_tiles = n_vals // n_reps
        n_reps //= n_coord_vals
        coord_idx = np.arange(n_coord_vals)
        coord_idx = np.tile(np.repeat(coord_idx, n_reps), n_tiles)
        idx[dim_axis] = coord_idx
        data[dim_axis] = coord.data[coord_idx]

    # Auxiliary coordinates.
    for aux_axis, coord in coords_copy.items():
        coord_idx = tuple(idx[dim[1:]] for dim in coord.dims)
        data[aux_axis] = coord.data[coord_idx]

    if omit_data:
        # NOTE: This is important when working with empty fields instead
        # directly with domains. The reason for not using domains might be
        # the fact that a domain does not contain the order of axes which
        # might be important.
        return data

    # Data.
    qty = darr.data
    vals = qty.magnitude
    match CopyApproach(allow_copy):
        case CopyApproach.NEVER:
            # This approach never copies an array. It returns a view if it
            # is possible to  change the shape in place. Otherwise, it
            # raises an `AttributeError`. See notes:
            # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
            # NOTE: This approach is not supported in `dask`.
            vals_ = vals.view()
            vals_.shape = (n_vals,)
        case CopyApproach.REQUIREMENT:
            # This approach returns a copy only if it is required to get
            # the array with a new shape. Otherwise it returns a view. See:
            # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
            # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html
            vals_ = vals.reshape(n_vals)
        case CopyApproach.CONTIGUOUS:
            # This approach returns a copy only if it is required to get
            # the contiguous array with a new shape. Otherwise it returns a
            # view. See:
            # https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
            vals_ = vals.ravel()
        case CopyApproach.ALWAYS:
            # This approach always copies an array. See:
            # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
            vals_ = vals.flatten()
    data[name] = pint.Quantity(vals_, qty.units)

    return data


def to_points_data_frame(
    name: str,
    dset: xr.Dataset,
    allow_copy: str | CopyApproach = "requirement",
    omit_data: bool = False,
) -> pd.DataFrame:
    # TODO: Check again whether this function makes copies.
    data = to_points_dict(name, dset, allow_copy, omit_data)
    data = {name: getattr(val, "magnitude", val) for name, val in data.items()}
    return pd.DataFrame(data=data, copy=False)
