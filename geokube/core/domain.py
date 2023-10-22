import abc
from typing import Mapping, Sequence, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis
from .coord_system import CoordinateSystem
from .quantity import create_quantity
from .feature import PointsFeature

class Domain():

    def _from_xrdset(self, ds: xr.Dataset) -> Self:
        return type(self)(
                coords={
                    axis_: coord.data
                    for axis_, coord in ds.coords.items()
                },
                coord_system=self.coord_system
            )

class Points(Domain, PointsFeature):

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_system: CoordinateSystem
    ) -> None:
        
        units = coord_system.units

        if isinstance(coords, Mapping):
            result_coords = {}
            n_pts = set()
            for axis_, vals in coords.items():
                vals_ = create_quantity(vals, units.get(axis_), axis_.dtype)
                if vals_.ndim != 1:
                    raise ValueError(
                        f"'coords' have axis {axis_} that does not have "
                        "one-dimensional values"
                    )
                result_coords[axis_] = vals_
                n_pts.add(vals_.size)
            if len(n_pts) != 1:
                raise ValueError("'coords' must have values of equal sizes")
            if not set(coord_system.axes) <= result_coords.keys():
                raise ValueError(
                    "'coords' must have all axes from the coordinate system"
                )
            self._n_points = n_pts.pop()
        elif isinstance(coords, Sequence):
            # NOTE: This approach currently does not allow providing units.
            n_dims = {len(point) for point in coords}
            if len(n_dims) != 1:
                raise ValueError(
                    "'coords' must have points of equal number of dimensions"
                )
            self._n_points = len(coords)
            data = pd.DataFrame(data=coords, columns=coord_system.axes)
            result_coords = {
                axis_: pint.Quantity(
                        vals.to_numpy(dtype=axis_.dtype), units.get(axis_)
                    )
                for axis_, vals in data.items()
            }
        else:
            raise TypeError("'coords' must be a sequence or mapping")

        super().__init__(coords=result_coords,
                         coord_system=coord_system)

class Profile(Domain):
    __slots__ = ('__n_prof', '__n_lev')

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        units = coord_system.units
        interm_coords = dict(coords)
        result_coords: dict[axis.Axis, xr.DataArray] = {}
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
            vert_ = create_quantity(
                vert, units.get(axis.vertical), axis.vertical.dtype
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
            vals_ = create_quantity(vals, units.get(axis_), axis_.dtype)
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
        if not set(coord_system.axes) <= result_coords.keys():
            raise ValueError(
                "'coords' must have all axes from the coordinate system"
            )
        self.__n_prof = n_prof_tot
        self.__n_lev = n_lev_tot
        super().__init__(domtype.Points, result_coords, coord_system)

    @property
    def number_of_profiles(self) -> int:
        return self.__n_prof

    @property
    def number_of_levels(self) -> int:
        return self.__n_lev


class Grid(Domain):
    # TODO: Consider auxiliary coordinates other than the
    # latitude and longitude. Especially consider how to represent them in the
    # API.
    # NOTE: The assumption is that the latitude and longitude as auxiliary
    # coordinates must have the dimensions either
    # `(axis.grid_latitude, axis.grid_longitude)` or `(axis.y, axis.x)`.
    __slots__ = ()

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        crs = coord_system.spatial.crs
        hor_dim_axes, hor_aux_axes = crs.dim_axes, crs.aux_axes
        hor_dims = tuple(f'_{axis_}' for axis_ in hor_dim_axes)
        dim_axes: tuple[str, ...]
        axes = set(coord_system.axes)
        if hor_aux_axes:
            axes -= set(hor_aux_axes)
        units = coord_system.units
        result_coords: dict[axis.Axis, xr.DataArray] = {}
        hor_aux_shapes = set()

        for axis_, vals in coords.items():
            vals_ = create_quantity(vals, units.get(axis_), axis_.dtype)
            if axis_ in axes:
                # Dimension coordinates.
                if not vals_.ndim:
                    dim_axes = ()
                elif vals_.ndim == 1:
                    dim_axes = (f'_{axis_}',)
                else:
                    raise ValueError(
                        f"'coords' have a dimension axis {axis_} that has "
                        "multi-dimensional values"
                    )
                # if not is_monotonic(vals_):
                #     raise ValueError(
                #         f"'coords' have a dimension axis {axis_} that does "
                #         "not have monotonic values"
                #     )
                # dim_axes = (axis_,)
            else:
                # Auxiliary coordinates.
                # dim_axes = hor_dim_axes
                dim_axes = hor_dims if vals_.ndim else ()
                if axis_ in hor_aux_axes:
                    hor_aux_shapes.add(vals_.shape)
            result_coords[axis_] = xr.DataArray(vals_, dims=dim_axes)
        if len(hor_aux_shapes) > 1:
            raise ValueError(
                "'coords' have auxiliary horizontal coordinates with different"
                "shapes"
            )
        super().__init__(result_coords, coord_system)
