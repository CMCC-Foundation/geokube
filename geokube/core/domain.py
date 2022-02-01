from __future__ import annotations

import functools as ft
import warnings
from collections.abc import Iterable
from enum import Enum
from itertools import chain
from typing import Any, Hashable, List, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, AxisType
from geokube.core.coord_system import (
    CoordSystem,
    CurvilinearGrid,
    RegularLatLon,
    parse_crs,
)
from geokube.utils import util_methods
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
from .coordinate import Coordinate, CoordinateType
from .enums import LatitudeConvention, LongitudeConvention
from .variable import Variable
from .domainmixin import DomainMixin


class DomainType(Enum):
    GRIDDED = "gridded"
    TIMESERIES = "timeseries"


class Domain(DomainMixin):

    __slots__ = (
        "_coords",
        "_crs",
        "_type",
        "_axis_to_name",
    )

    _LOG = HCubeLogger(name="Domain")

    def __init__(
        self,
        coords: Union[
            Mapping[Hashable, Tuple[np.ndarray, ...]], Iterable[Coordinate], Domain
        ],
        crs: CoordSystem,
        domaintype: Optional[DomainType] = None,
    ) -> None:
        if isinstance(coords, dict):
            self._coords = {}
            for name, coord in coords.items():
                self._coords[name] = Domain._as_coordinate(coord, name)
        if isinstance(coords, list):
            # TODO: check if it is a coordinate or just data!
            self._coords = {c.name: c for c in coords}
        if isinstance(coords, Domain):
            self._coords = coords._coords
            self._crs = coords._crs
            self._type = coords._type
            self._axis_to_name = coords._axis_to_name

        self._crs = crs
        self._type = domaintype
        self._axis_to_name = {c.axis_type: c.name for c in self._coords.values()}

    @classmethod
    def _as_coordinate(cls, coord, name) -> Coordinate:
        if isinstance(coord, Coordinate):
            return coord
        elif isinstance(coord, tuple):
            # tupl -> (data, dims, axis)
            return Coordinate(data=coord[0], dims=coord[1], axis=coord[2])
        else:
            return Coordinate(data=coord, axis=name)

    @property
    def type(self):
        return self._type

    @property
    def coords(self):
        return self._coords

    @property
    def crs(self) -> CoordSystem:  # horizontal coordinate reference system
        return self._crs

    @property
    def aux_coords(self) -> List[str]:
        return [c.name for c in self._coords.values() if not c.is_dim]

    def __repr__(self) -> str:
        return self.to_xarray(encoding=False).__repr__()

    #        return formatting.array_repr(self.to_xarray())

    def _repr_html_(self):
        return self.to_xarray(encoding=False)._repr_html_()
        # if OPTIONS["display_style"] == "text":
        #     return f"<pre>{escape(repr(self.to_xarray()))}</pre>"
        # return formatting_html.array_repr(self)

    def __eq__(self, other):
        if self.crs != other.crs:
            return False
        coord_keys_eq = set(self._coords.keys()) == set(other._coords.keys())
        if not coord_keys_eq:
            return False
        for ck in self._coords.keys():
            if self._coords[ck].axis_type is Axis.TIME:
                if not np.all(self._coords[ck].values == other._coords[ck].values):
                    return False
            else:
                if not np.allclose(self._coords[ck].values, other._coords[ck].values):
                    return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __len__(self) -> int:
        return len(self._coords)

    def __setitem__(self, key: str, value: Union[Coordinate, Variable]):
        self._coords[key] = value

    def __contains__(self, key: str):
        return (key in self._coords) or (AxisType.parse(key) in self._axis_to_name)

    def nbytes(self) -> int:
        return sum(coord.nbytes for coord in self._coords)

    @log_func_debug
    def _process_time_combo(self, indexer: Mapping[Hashable, Any]):
        if "time" in indexer:
            indexer = indexer["time"]

        def _reduce_boolean_selection(_ds, key: str, _time_indexer):
            if key in _time_indexer.keys():
                dt = getattr(_ds, key).values
                XX = ft.reduce(
                    lambda x, y: x | (dt == y),
                    [False] + list(np.array(_time_indexer[key], dtype=int, ndmin=1)),
                )
                return XX
            return True

        if (time_coord := self[Axis.TIME]) is None:
            raise ex.HCubeNoSuchAxisError(
                f"Time axis was not found for that dataset!", logger=self._LOG
            )
        time_coord = time_coord.to_xarray_dataarray()
        time_coord_dt = time_coord.dt

        year_mask = _reduce_boolean_selection(time_coord_dt, "year", indexer)
        month_mask = _reduce_boolean_selection(time_coord_dt, "month", indexer)
        day_mask = _reduce_boolean_selection(time_coord_dt, "day", indexer)
        hour_mask = _reduce_boolean_selection(time_coord_dt, "hour", indexer)
        inds = np.where(year_mask & month_mask & day_mask & hour_mask)[0]
        inds = util_methods.list_to_slice_or_array(inds)
        return {time_coord.name: inds}

    @log_func_debug
    def compute_bounds(self, coord, force: bool = False) -> None:
        # check if coord is Latitude or Longitude or raise an error
        coord = self[coord]
        if coord.ctype is not CoordinateType.INDEPENDENT:
            raise ex.HCubeValueError(
                f"Calculating bounds is supported only for independent coordinate, but requested coordinate has type: {coord.ctype}",
                logger=self._LOG,
            )
        # Handling the case when bounds already exist, according to `force`
        if coord.bounds is not None:
            msg = f"{coord.name} bounds already exist"
            if not force:
                warnings.warn(f"{msg} and are not going be modified")
                self._LOG.warn(f"{msg} and are not going be modified")
                return
            warnings.warn(f"{msg} and are going to be recalculated")
            self._LOG.warn(f"{msg} and are going to be recalculated")

        # Handling the case when `crs` is `None` or not instance of `GeogCS`
        crs = self._crs
        if crs is None:
            raise ex.HCubeValueError(
                "'crs' is None and cell bounds cannot be calculated", logger=self._LOG
            )
        if not isinstance(crs, RegularLatLon):
            raise ex.HCubeNotImplementedError(
                f"'{crs.__class__.__name__}' is currently not supported for "
                "calculating cell corners",
                logger=self._LOG,
            )

        # Calculating bounds
        val = coord.data
        val_b = np.empty(shape=val.size + 1, dtype=np.float64)
        val_b[1:-1] = 0.5 * (val[:-1] + val[1:])
        half_step = 0.5 * (val.ptp() / (val.size - 1))
        # The case `val[0] > val[-1]` represents reversed order of values:
        i, j = (0, -1) if val[0] <= val[-1] else (-1, 0)
        val_b[i] = val[i] - half_step
        val_b[j] = val[j] + half_step
        # Making sure that longitude and latitude values are not outside their
        # ranges
        range_b = ()
        if coord.axis.atype == Axis.LONGITUDE:
            if self.longitude_convention is LongitudeConvention.POSITIVE_WEST:
                range_b = (0.0, 360.0)
            else:
                range_b = (-180.0, 180.0)
        elif coord.axis.atype == Axis.LATITUDE:
            range_b = (-90.0, 90.0)

        if range_b:
            val_b[i] = val_b[i].clip(*range_b)
            val_b[j] = val_b[j].clip(*range_b)

        # Bounds are stored as 1D array of size (coord_vals.shape + 1)
        # It needs to be stored as array of shape (len(coord_vals), 2)
        # Setting `coordinate.bounds`
        name = f"{coord.name}_bnds"
        coord.bounds = Variable(
            name=name,
            data=Domain.convert_bounds_1d_to_2d(val_b),
            units=coord.units,
            dims=(coord.dims[0].name, "bounds"),
        )

    @staticmethod
    def convert_bounds_1d_to_2d(values):
        assert values.ndim == 1
        return np.vstack((values[:-1], values[1:])).T

    @staticmethod
    def convert_bounds_2d_to_1d(values):
        assert values.ndim == 2
        return np.concatenate((values[:, 0], values[[-1], 1]))

    @classmethod
    def guess_crs(cls, da: Union[xr.Dataset, xr.DataArray]):
        # TODO: implement more logic
        if "nav_lat" in da.coords or "nav_lon" in da.coords:
            return CurvilinearGrid()
        return RegularLatLon()

    @classmethod
    @log_func_debug
    def merge(cls, domains: List[Domain]):
        # check if the domains are defined on the same crs
        coords = {}
        for domain in domains:
            coords.update(**domain.coords)
        return Domain(coords=coords, crs=domains[0].crs)

    @classmethod
    @log_func_debug
    def from_xarray(
        cls,
        ds: xr.Dataset,
        ncvar: str,
        id_pattern: str = None,
        copy: bool = False,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> "Domain":

        da = ds[ncvar]
        coords = []

        for dim in da.dims:
            if dim in da.coords:
                coords.append(
                    Coordinate.from_xarray(
                        ds=ds, ncvar=dim, id_pattern=id_pattern, mapping=mapping
                    )
                )

        xr_coords = ds[ncvar].attrs.get(
            "coordinates", ds[ncvar].encoding.get("coordinates", None)
        )
        if xr_coords is not None:
            for coord in xr_coords.split(" "):
                coords.append(
                    Coordinate.from_xarray(
                        ds=ds, ncvar=coord, id_pattern=id_pattern, mapping=mapping
                    )
                )
        if "grid_mapping" in da.encoding:
            crs = parse_crs(da[da.encoding.get("grid_mapping")])
        elif "grid_mapping" in da.attrs:
            crs = parse_crs(da[da.attrs.get("grid_mapping")])
        else:
            crs = Domain.guess_crs(da)

        return Domain(coords=coords, crs=crs)

    @log_func_debug
    def to_xarray(self, encoding=True) -> xr.core.coordinates.DatasetCoordinates:
        grid = {}
        for coord in self._coords.values():
            if encoding:
                coord_name = coord.ncvar
            else:
                coord_name = coord.name
            grid[coord_name] = coord.to_xarray(encoding)  # to xarray variable
            if (bounds := coord.bounds) is not None:
                continue
                # TODO: bounds support latter
                if len(bounds) > 1:
                    raise ex.HCubeNotImplementedError(
                        f"Multiple bounds are currently not supported!"
                    )
                for bnd in bounds.values():
                    if encoding:
                        bounds_name = bnd.ncvar
                    else:
                        bounds_name = bnd.name
                    grid[bounds_name] = bnd.to_xarray(encoding)  # to xarray variable

        if self.crs is not None:
            not_none_attrs = self.crs.as_crs_attributes()
            not_none_attrs["grid_mapping_name"] = self.crs.grid_mapping_name
            grid["crs"] = xr.DataArray(1, name="crs", attrs=not_none_attrs)
        return xr.Dataset(coords=grid).coords
