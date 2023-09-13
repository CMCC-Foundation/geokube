from typing import Mapping, Sequence

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis
from .coord_system import CoordinateSystem


class Points:
    __slots__ = ('__coords', '__coord_system', '__n_pts')

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coord_system, CoordinateSystem):
            raise TypeError(
                "'coord_system' must be an instance of 'CoordinateSystem'"
            )
        self.__coord_system = coord_system

        units = coord_system.units
        pts = ('_points',)

        if isinstance(coords, Mapping):
            result_coords = {}
            n_pts = set()
            for axis_, vals in coords.items():
                vals_ = _create_quantity(vals, units.get(axis_), axis_.dtype)
                if vals_.ndim != 1:
                    raise ValueError(
                        f"'coords' have axis {axis_} that does not have "
                        "one-dimensional values"
                    )
                result_coords[axis_] = xr.DataArray(vals_, dims=pts)
                n_pts.add(vals_.size)
            if len(n_pts) != 1:
                raise ValueError("'coords' must have values of equal sizes")
            if not set(self.__coord_system.axes) <= result_coords.keys():
                raise ValueError(
                    "'coords' must have all axes from the coordinate system"
                )
            self.__n_pts = n_pts.pop()
            self.__coords = result_coords
        elif isinstance(coords, Sequence):
            # NOTE: This approach currently does not allows providing units.
            n_dims = {len(point) for point in coords}
            if len(n_dims) != 1:
                raise ValueError(
                    "'coords' must have points of equal number of dimensions"
                )
            self.__n_pts = len(coords)
            data = pd.DataFrame(data=coords, columns=coord_system.axes)
            self.__coords = {
                axis_: xr.DataArray(
                    pint.Quantity(
                        vals.to_numpy(dtype=axis_.dtype), units.get(axis_)
                    ),
                    dims=pts
                )
                for axis_, vals in data.items()
            }
        else:
            raise TypeError("'coords' must be a sequence or mapping")

    @property
    def coord_system(self) -> CoordinateSystem:
        return self.__coord_system

    @property
    def _coords(self) -> dict[axis.Axis, xr.DataArray]:
        return self.__coords

    @property
    def number_of_points(self) -> int:
        return self.__n_pts

    def coordinates(
        self, coord_axis: axis.Axis | None = None
    ) -> dict[axis.Axis, pint.Quantity] | pint.Quantity:
        if coord_axis is None:
            return {axis: coord.data for axis, coord in self.__coords.items()}
        return self.__coords[coord_axis].data


class Profile:
    __slots__ = ('__coords', '__coord_system', '__n_prof', '__n_lev')

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coord_system, CoordinateSystem):
            raise TypeError(
                "'coord_system' must be an instance of 'CoordinateSystem'"
            )
        self.__coord_system = coord_system

        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        units = coord_system.units
        interm_coords = dict(coords)
        result_coords = {}
        prof = ('_profiles',)
        n_prof = set()
        vert = interm_coords.pop(axis.vertical)

        # Vertical.
        if isinstance(vert, Sequence):
            n_lev = [len(vert_val) for vert_val in vert]
            n_lev_tot = max(n_lev)
            n_prof_tot = len(vert)
            vert_vals = np.empty(
                shape=(n_prof_tot, n_lev_tot), dtype=axis.vertical.dtype
            )
            for i, (stop_idx, vals) in enumerate(zip(n_lev, vert)):
                if stop_idx == n_lev_tot:
                    vert_vals[i, :] = vals
                else:
                    vert_vals[i, :stop_idx] = vals
                    vert_vals[i, stop_idx:] = np.nan
            vert_ = pint.Quantity(vert_vals, units[axis.vertical])
        else:
            vert_ = _create_quantity(
                vert,
                units.get(axis.vertical, axis.vertical.dtype),
                axis.vertical.dtype
            )
            vert_shape = vert_.shape
            if len(vert_shape) != 2:
                raise ValueError(
                    "'coords' must have vertical as a two-dimensional data "
                    "structure"
                )
            n_prof_tot, n_lev_tot = vert_shape
        result_coords[axis.vertical] = xr.DataArray(
            vert_, dims=('_profiles', '_levels')
        )

        # All coordinates except the vertical.
        for axis_, vals in interm_coords.items():
            vals_ = _create_quantity(vals, units.get(axis_), axis_.dtype)
            if vals_.ndim != 1:
                raise ValueError(
                    f"'coords' have axis {axis_} that does not have "
                    "one-dimensional values"
                )
            result_coords[axis_] = xr.DataArray(vals_, dims=prof)
            n_prof.add(vals_.size)
        if len(n_prof) != 1:
            raise ValueError(
                "'coords' with the exception of vertical must have values of "
                "equal sizes"
            )
        if n_prof_tot != n_prof.pop():
            raise ValueError("'coords' have items of with inappropriate sizes")
        if not set(self.__coord_system.axes) <= result_coords.keys():
            raise ValueError(
                "'coords' must have all axes from the coordinate system"
        )
        self.__n_prof = n_prof_tot
        self.__n_lev = n_lev_tot
        self.__coords = result_coords

    @property
    def coord_system(self) -> CoordinateSystem:
        return self.__coord_system

    @property
    def _coords(self) -> dict[axis.Axis, xr.DataArray]:
        return self.__coords

    @property
    def number_of_profiles(self) -> int:
        return self.__n_prof

    @property
    def number_of_levels(self) -> int:
        return self.__n_lev

    def coordinates(
        self, coord_axis: axis.Axis | None = None
    ) -> dict[axis.Axis, pint.Quantity] | pint.Quantity:
        if coord_axis is None:
            return {axis: coord.data for axis, coord in self.__coords.items()}
        return self.__coords[coord_axis].data


def _create_quantity(
    values: npt.ArrayLike | pint.Quantity,
    default_units: pint.Unit | None,
    default_dtype: np.dtype
) -> pint.Quantity:
    match values:
        case pint.Quantity() if isinstance(values.magnitude, np.ndarray):
            return values
        case pint.Quantity():
            return pint.Quantity(np.asarray(values.magnitude), values.units)
        case np.ndarray():
            # NOTE: The pattern arr * unit does not work when arr has stings.
            return pint.Quantity(values, default_units)
        case da.Array():
            return pint.Quantity(values.compute(), default_units)
        case _:
            return pint.Quantity(
                np.asarray(values, dtype=default_dtype), default_units
            )
