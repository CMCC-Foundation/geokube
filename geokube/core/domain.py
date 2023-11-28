from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import pint_xarray
import xarray as xr
from pyproj import Transformer

from . import axis
from .coord_system import CoordinateSystem
from .feature import GridFeature, PointsFeature, ProfilesFeature
from .quantity import get_magnitude, create_quantity
from .crs import Geodetic
from .units import units


# TODO: Consider renaming this class to `DomainMixin`.
# NOTE: maybe we don't need this class
class Domain:
    __slots__ = ()

    @classmethod
    def _from_xrdset(
        cls, dset: xr.Dataset, coord_system: CoordinateSystem
    ) -> Self:
        return cls(coords=dset.coords, coord_system=coord_system)

    @classmethod
    def as_xarray_dataset(
        cls,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity | xr.DataArray],
        coord_system: CoordinateSystem
    ) -> xr.Dataset:
        da = coord_system.crs.as_xarray()
        r_coords = coords
        r_coords[da.name] = da
        ds = xr.Dataset(
            coords=r_coords
        )
        ds.attrs['grid_mapping'] = da.name
        return ds

class Points(PointsFeature):
    __slots__ = ()

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_system: CoordinateSystem
    ) -> None:
        units = coord_system.units

        match coords:
            case Mapping():
                result_coords = {}
                for axis_, coord in coords.items():
                    result_coords[axis_] = qty = create_quantity(
                        coord, units.get(axis_), axis_.encoding['dtype']
                    )
                    if qty.ndim != 1:
                        raise ValueError(
                            f"'coords' have axis {axis_} that does not have "
                            "one-dimensional values"
                        )
                if not set(coord_system.axes) <= result_coords.keys():
                    raise ValueError(
                        "'coords' must have all axes contained in the "
                        "coordinate system"
                    )
            case Sequence():
                # NOTE: This approach currently does not allow providing units.
                n_dims = {len(point) for point in coords}
                if len(n_dims) != 1:
                    raise ValueError(
                        "'coords' must have points of equal number of "
                        "dimensions"
                    )
                data = pd.DataFrame(data=coords, columns=coord_system.axes)
                result_coords = {
                    axis_: pint.Quantity(
                        vals.to_numpy(dtype=axis_.encoding['dtype']),
                        units.get(axis_)
                    )
                    for axis_, vals in data.items()
                }
            case _:
                raise TypeError("'coords' must be a sequence or mapping")

        ds = Domain.as_xarray_dataset(result_coords, coord_system)

        super().__init__(ds=ds)


class Profiles(Domain, ProfilesFeature):
    __slots__ = ()

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
        prof = (self._DIMS_[0],)
        n_prof = set()
        vert = interm_coords.pop(axis.vertical)

        # Vertical.
        if isinstance(vert, Sequence):
            n_lev = [len(vert_val) for vert_val in vert]
            n_lev_tot = max(n_lev)
            n_prof_tot = len(vert)
            vert_vals = np.empty(
                shape=(n_prof_tot, n_lev_tot),
                dtype=axis.vertical.encoding['dtype']
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
                vert, units.get(axis.vertical), axis.vertical.encoding['dtype']
            )
            vert_shape = vert_.shape
            if len(vert_shape) != 2:
                raise ValueError(
                    "'coords' must have vertical as a two-dimensional data "
                    "structure"
                )
            n_prof_tot, n_lev_tot = vert_shape

        result_coords[axis.vertical] = xr.DataArray(
            vert_, dims=self._DIMS_
        )

        # All coordinates except the vertical.
        for axis_, vals in interm_coords.items():
            qty = create_quantity(
                vals, units.get(axis_), axis_.encoding['dtype']
            )
            if qty.ndim != 1:
                raise ValueError(
                    f"'coords' have axis {axis_} that does not have "
                    "one-dimensional values"
                )
            result_coords[axis_] = xr.DataArray(qty, dims=prof)
            n_prof.add(qty.size)
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

        super().__init__(coords=result_coords, coord_system=coord_system)


class Grid(Domain, GridFeature):
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
        units = coord_system.units
        interm_coords = dict(coords)

        # TODO: REVIEW! - we need to keep the order as in the CRS
        result_coords = {
            axis_: create_quantity(
                values=interm_coords.pop(axis_),
                default_units=units.get(axis_),
                default_dtype=axis_.encoding['dtype']
            )
            for axis_ in coord_system.dim_axes
        }
        result_coords |= {
            axis_: create_quantity(
                values=coord,
                default_units=units.get(axis_),
                default_dtype=axis_.encoding['dtype']
            )
            for axis_, coord in interm_coords.items()
        }

        super().__init__(result_coords, coord_system)

    def infer_resolution(self, axis):
        return self.coords[axis].ptp() / (self.coords[axis].size - 1)

    def as_geodetic(self, as_points=False):
        coord_system = CoordinateSystem(
            horizontal=Geodetic(),
            elevation = self.coord_system.elevation,
            time = self.coord_system.time,
            user_axes = self.coord_system.user_axes
        )

        # Infering latitude and longitude steps from the x and y coordinates.
        # this works only for Geodetic and Rotated Pole
        # It should be generalized also for projections
        # TODO: once we get the resolution for the horizontal we should transform
        # in a value in lat/lon 
        lat_step = self.infer_resolution(self.crs.dim_Y_axis)
        lon_step = self.infer_resolution(self.crs.dim_X_axis)

        # Building regular latitude-longitude coordinates.
        lat_vals = self.coords[axis.latitude]
        lon_vals = self.coords[axis.longitude]
        south, north = lat_vals.min().magnitude, lat_vals.max().magnitude
        west, east = lon_vals.min().magnitude, lon_vals.max().magnitude
#        lat = np.arange(south, north + lat_step / 2, lat_step)
#        lon = np.arange(west, east + lon_step / 2, lon_step)
        lat = np.arange(south, north, lat_step)
        lon = np.arange(west, east, lon_step)
        
        coords = self.coords  # Or `self.coords.copy()`.
        for axis_ in self.crs.axes:
            del coords[axis_]
        hor_coords = {
            coord_system.crs.dim_Y_axis: lat,
            coord_system.crs.dim_X_axis: lon
        }
        coords |= hor_coords

        return type(self)(coords=coords, coord_system=coord_system)

    def spatial_transform_to(self, crs):
        # TODO: we assume that they have the same datum. We need to change
        # the code when datum are different!!
        # 
        lat = get_magnitude(self.coords[axis.latitude], units['degrees_N'])
        lon = get_magnitude(self.coords[axis.longitude], units['degrees_E'])
        if lat.ndim == lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        transformer = Transformer.from_crs(
            crs_from=self.crs._crs,
            crs_to=crs._crs,
            always_xy=True
        )
        # x, y = transformer.transform(lon, lat)
        return transformer.transform(lon, lat)
