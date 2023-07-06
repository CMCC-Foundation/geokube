from __future__ import annotations

import functools as ft
from numbers import Number
import warnings
from collections.abc import Iterable
from enum import Enum
from itertools import chain
from statistics import mode
from typing import Any, Hashable, List, Mapping, Optional, Tuple, Union

import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr

from ..utils import util_methods
from ..utils.decorators import geokube_logging
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis, AxisType
from .coord_system import (
    CoordSystem,
    CurvilinearGrid,
    GeogCS,
    RegularLatLon,
    parse_crs,
    add_prefix,
    trim_prefix
)
from .coordinate import Coordinate, CoordinateType
from .domainmixin import DomainMixin
from .enums import LatitudeConvention, LongitudeConvention
from .variable import Variable

_COORDS_TUPLE_CONTENT = [
    "dims",
    "data",
    "bounds",
    "units",
    "properties",
    "encoding",
]


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class DomainType(Enum):
    GRIDDED = "gridded"
    POINTS = "points"
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
            Mapping[Hashable, Tuple[np.ndarray, ...]],
            Iterable[Coordinate],
            Domain,
        ],
        crs: CoordSystem,
        domaintype: Optional[DomainType] = None,
    ) -> None:
        if isinstance(coords, dict):
            self._coords = {}
            for name, coord in coords.items():
                self._coords[name] = Domain._as_coordinate(coord, name)
        if isinstance(coords, (list, set)):
            # TODO: check if it is a coordinate or just data!
            self._coords = {c.name: c for c in coords}
        if isinstance(coords, Domain):
            self._coords = coords._coords
            self._crs = coords._crs
            self._type = coords._type
            self._axis_to_name = coords._axis_to_name

        self._crs = crs
        self._type = domaintype
        self._axis_to_name = {
            c.axis_type: c.name for c in self._coords.values()
        }

    @classmethod
    def _as_coordinate(cls, coord, name) -> Coordinate:
        if isinstance(coord, Coordinate):
            return coord
        elif isinstance(coord, tuple):
            # tupl -> (data, dims, axis)
            l = dict(enumerate(coord))
            return Coordinate(
                data=coord[0], dims=l.get(1, name), axis=l.get(2, name)
            )
        else:
            return Coordinate(data=coord, axis=name)
    
    @property
    def grid_mapping_name(self) -> str:
        return add_prefix(self.crs.grid_mapping_name)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def coords(self):
        return self._coords

    @property
    def crs(self) -> CoordSystem:  # horizontal coordinate reference system
        return self._crs

    @crs.setter
    def crs(self, crs):
        self._crs = crs

    @property
    def aux_coords(self) -> List[str]:
        return [c.name for c in self._coords.values() if not c.is_dim]

    def _infer_resolution(self):
        grid_x = np.abs(mode(np.diff(self.longitude)))
        grid_y = np.abs(mode(np.diff(self.latitude)))
        return (grid_x, grid_y)

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
            if self._coords[ck].axis_type is AxisType.TIME:
                if not np.all(
                    self._coords[ck].values == other._coords[ck].values
                ):
                    return False
            else:
                if not np.allclose(
                    self._coords[ck].values, other._coords[ck].values
                ):
                    return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __len__(self) -> int:
        return len(self._coords)

    def __setitem__(self, key: str, value: Union[Coordinate, Variable]):
        self._coords[key] = value

    def __contains__(self, key: Union[str, Axis, AxisType]) -> bool:
        if isinstance(key, Axis):
            return key.type in self._axis_to_name
        return (key in self._coords) or (
            AxisType.parse(key) in self._axis_to_name
        )

    def __next__(self):
        for k, v in self._coords.items():
            yield k, v
        raise StopIteration

    @property
    def nbytes(self) -> int:
        return sum(coord.nbytes for coord in self._coords.values())

    def map_indexers(self, indexers: Mapping[str, Any]) -> Mapping[Axis, Any]:
        return {Axis(n): v for n, v in indexers.items()}

    @geokube_logging
    def _process_time_combo(self, indexer: Mapping[Hashable, Any]):
        if "time" in indexer:
            indexer = indexer["time"]

        def _reduce_boolean_selection(_ds, key: str, _time_indexer):
            if key in _time_indexer.keys():
                dt = getattr(_ds, key).values
                XX = ft.reduce(
                    lambda x, y: x | (dt == y),
                    [False]
                    + list(np.array(_time_indexer[key], dtype=int, ndmin=1)),
                )
                return XX
            return True

        if (time_coord := self[AxisType.TIME]) is None:
            raise KeyError(f"Time axis was not found for that dataset!")
        time_coord_dset = time_coord.to_xarray(encoding=False)
        time_coord_dt = time_coord_dset[time_coord.name].dt

        year_mask = _reduce_boolean_selection(time_coord_dt, "year", indexer)
        month_mask = _reduce_boolean_selection(time_coord_dt, "month", indexer)
        day_mask = _reduce_boolean_selection(time_coord_dt, "day", indexer)
        hour_mask = _reduce_boolean_selection(time_coord_dt, "hour", indexer)
        inds = np.where(year_mask & month_mask & day_mask & hour_mask)[0]
        inds = util_methods.list_to_slice_or_array(inds)
        return {time_coord.name: inds}

    @geokube_logging
    def compute_bounds(
        self, coordinate: str | None = None, force: bool = False
    ) -> None:
        # Defining behavior if `coordinate` is different from `'latitude'` or
        # `'longitude'`
        if coordinate is None:
            self.compute_bounds(coordinate="latitude", force=False)
            self.compute_bounds(coordinate="longitude", force=False)
            return
        elif coordinate not in {"latitude", "longitude"}:
            raise NotImplementedError(
                "'coordinate' must be either 'latitude' or 'longitude', other "
                "values are not currently supported"
            )

        # Extracting the coordinate object
        coord = self[coordinate]

        # Handling the case when bounds already exist, according to `force`
        if coord.bounds is not None:
            msg = f"{coord.name} bounds already exist"
            if not force:
                warnings.warn(f"{msg} and are not going be modified")
                self._LOG.warn(f"{msg} and are not going be modified")
                return
            warnings.warn(f"{msg} and are going to be recalculated")
            self._LOG.warn(f"{msg} and are going to be recalculated")

        # Cases for dependent or scalar coordinates are not handled
        if coord.type is not CoordinateType.INDEPENDENT:
            raise NotImplementedError(
                "'coordinate' must be independent to calculate its bounds, "
                "dependent coordinates are not currently supported"
            )

        # Handling the case when `crs` is `None` or not instance of `GeogCS`
        crs = self._crs
        if crs is None:
            # TODO: Reconsider if this should be `ValueError` or some other
            # type of exception, see:
            # https://docs.python.org/3/library/exceptions.html#ValueError
            raise ValueError(
                "'crs' is None and cell bounds cannot be calculated"
            )
        if not isinstance(crs, GeogCS):
            raise NotImplementedError(
                f"'{crs.__class__.__name__}' is currently not supported for "
                "calculating cell corners"
            )

        # Calculating bounds
        val = coord.values
        val_b = np.empty(shape=val.size + 1, dtype=np.float64)
        val_b[1:-1] = 0.5 * (val[:-1] + val[1:])
        half_step = 0.5 * (val.ptp() / (val.size - 1))
        # The case `val[0] > val[-1]` represents reversed order of values:
        i, j = (0, -1) if val[0] <= val[-1] else (-1, 0)
        val_b[i] = val[i] - half_step
        val_b[j] = val[j] + half_step
        # Making sure that longitude and latitude values are not outside their
        # ranges
        if coord.axis_type is AxisType.LONGITUDE:
            if coord.convention is LongitudeConvention.POSITIVE_WEST:
                range_b = (0.0, 360.0)
            else:
                range_b = (-180.0, 180.0)
        else:  # Case when coord.axis_type is AxisType.LATITUDE
            range_b = (-90.0, 90.0)
        val_b[i] = val_b[i].clip(*range_b)
        val_b[j] = val_b[j].clip(*range_b)

        # Setting `coordinate.bounds`
        coord.bounds = Domain.convert_bounds_1d_to_2d(val_b)

    @geokube_logging
    def _calculate_missing_lat_and_lon(self):
        # NOTE: This approach is elegant and accurate, but it issues warnings
        # because of the `@geokube_logging` decorator.
        # missing_lat_or_lon = False
        # try:
        #     self.latitude, self.longitude
        # except KeyError:
        #     missing_lat_or_lon = True
        missing_lat_or_lon = not (
            {AxisType.LATITUDE, AxisType.LONGITUDE}
            <= self._axis_to_name.keys()
        )

        # TODO: Consider moving these checks to `Field.to_xarray`.
        if (
            missing_lat_or_lon
            and (self._type is DomainType.GRIDDED or self._type is None)
            and self._crs is not None
            and not isinstance(self._crs, GeogCS)
            and self.x.type is CoordinateType.INDEPENDENT
            and self.y.type is CoordinateType.INDEPENDENT
        ):
            domain_x, domain_y = self.x, self.y
            dims = (domain_y.dims[0], domain_x.dims[0])
            x, y = np.meshgrid(domain_x.to_numpy(), domain_y.to_numpy())
            pts = (
                GeogCS(6371229)
                .as_cartopy_crs()
                .transform_points(src_crs=self._crs.as_cartopy_crs(), x=x, y=y)
            )
            lon_vals, lat_vals = pts[..., 0], pts[..., 1]
            lat_axis = Axis(
                name="latitude", axistype=AxisType.LATITUDE, is_dim=False
            )
            lon_axis = Axis(
                name="longitude", axistype=AxisType.LONGITUDE, is_dim=False
            )
            lat_coord = Coordinate(
                data=lat_vals, axis=lat_axis, dims=dims, units="degree_north"
            )
            lon_coord = Coordinate(
                data=lon_vals, axis=lon_axis, dims=dims, units="degree_east"
            )
            new_coords = {"latitude": lat_coord, "longitude": lon_coord}
            self._coords.update(new_coords)
            new_names = {
                AxisType.LATITUDE: "latitude",
                AxisType.LONGITUDE: "longitude",
            }
            self._axis_to_name.update(new_names)

    @staticmethod
    def convert_bounds_1d_to_2d(values):
        assert values.ndim == 1
        return np.vstack((values[:-1], values[1:])).T

    @staticmethod
    def convert_bounds_2d_to_1d(values):
        assert values.ndim == 2
        return np.concatenate((values[:, 0], values[[-1], 1]))

    @classmethod
    def guess_crs(
        cls,
        da: Union[xr.Dataset, xr.DataArray, Mapping[str, Coordinate]],
    ):
        # TODO: implement more logic
        if isinstance(da, (xr.Dataset, xr.DataArray)):
            if "nav_lat" in da.coords or "nav_lon" in da.coords:
                return CurvilinearGrid()
        if isinstance(da, dict):
            if "nav_lat" in da or "nav_lon" in da:
                return CurvilinearGrid()
        return GeogCS(6371229)

    @classmethod
    @geokube_logging
    def merge(cls, domains: List[Domain]):
        # TODO: check if the domains are defined on the same crs
        coords = {}
        domaintype = None
        for domain in domains:
            coords.update(**domain.coords)
            domaintype = domain.type
        return Domain(coords=coords, crs=domains[0].crs, domaintype=domaintype)

    @classmethod
    @geokube_logging
    def from_xarray(
        cls,
        ds: xr.Dataset,
        ncvar: str,
        id_pattern: str = None,
        copy: bool = False,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> "Domain":
        da = ds[ncvar]
        coords = set()
        for dim_name in da.dims:
            if dim_name in da.coords:
                coords.add(
                    Coordinate.from_xarray(
                        ds=ds,
                        ncvar=dim_name,
                        id_pattern=id_pattern,
                        mapping=mapping,
                    )
                )
        xr_coords = ds[ncvar].attrs.get(
            "coordinates", ds[ncvar].encoding.get("coordinates", None)
        )
        if xr_coords is not None:
            for coord_name in xr_coords.split(" "):
                if coord_name not in ds:
                    warnings.warn(
                        f"Coordinate {coord_name} does not exist in the"
                        " dataset!"
                    )
                    continue
                coord = Coordinate.from_xarray(
                    ds=ds,
                    ncvar=coord_name,
                    id_pattern=id_pattern,
                    mapping=mapping,
                )
                if coord in coords:
                    warnings.warn(
                        f"Coordinate {coord_name} was already defined as"
                        " dimension!"
                    )
                    continue
                coords.add(coord)
        if "grid_mapping" in da.encoding:
            crs = parse_crs(da[da.encoding.get("grid_mapping")])
        elif "grid_mapping" in da.attrs:
            crs = parse_crs(da[da.attrs.get("grid_mapping")])
        else:
            crs = Domain.guess_crs(da)

        # NOTE: a workaround for keeping domaintype
        # ds attributes are modified!
        # Issue: https://github.com/geokube/geokube/issues/147
        if (
            domain_type := ds[ncvar].attrs.pop("__geo_domtype", None)
        ) is not None:
            return Domain(
                coords=coords, crs=crs, domaintype=DomainType(domain_type)
            )
        return Domain(coords=coords, crs=crs)

    @geokube_logging
    def to_xarray(
        self, encoding=True
    ) -> xr.core.coordinates.DatasetCoordinates:
        grid = {}
        grid = xr.Dataset().coords
        for coord in self._coords.values():
            grid.update(coord.to_xarray(encoding=encoding))

        if self.crs is not None:
            crs_name = self.grid_mapping_name
            not_none_attrs = self.crs.as_crs_attributes()
            # NOTE: we keep cf-compliant value of `grid_mapping_name` attribute
            not_none_attrs["grid_mapping_name"] = self.crs.grid_mapping_name
            grid.update(
                {
                    crs_name: xr.DataArray(
                        1, name=crs_name, attrs=not_none_attrs
                    )
                }
            )
        return grid

    def to_dict(self, unique_values=False):
        return {
            "crs": self._crs.to_dict(),
            "coordinates": {
                name: coord.to_dict(unique_values)
                for name, coord in self._coords.items()
            },
        }

    @classmethod
    def _make_domain_from_coords_dict_dims_and_crs(
        cls, coords, dims, crs=None
    ):
        """Return a domain based on coords dict, dims, and coordinate reference system.

        coords can be in the form {"latitude": lat_value} or in the form where the value
        is a tuple. That tuple might contain following elements:
        (dims: tuple[str], data, unit: optional, bounds: optional, properties: optional, encoding: optional)

        """
        if not isinstance(coords, dict):
            raise TypeError(
                f"Expected type of `coords` is `dict`, but `{type(coords)}`"
                " provided!"
            )
        res_coords = []
        for k, v in coords.items():
            if isinstance(v, pd.core.indexes.datetimes.DatetimeIndex):
                v = np.array(v)
            if isinstance(v, (Number, np.ndarray, da.Array)):
                # If coord provided not as tuple, be default it is deemed as `dimension`
                if isinstance(k, AxisType):
                    axis = Axis(name=k.axis_type_name, axistype=k, is_dim=True)
                elif isinstance(k, (Axis, str)):
                    axis = Axis(k, is_dim=True)
                res_coords.append(Coordinate(data=v, axis=axis, dims=(axis,)))
            elif isinstance(v, tuple):
                dims = data = bounds = units = props = encoding = None
                if len(v) >= 1:
                    dims = v[0]
                if len(v) >= 2:
                    data = v[1]
                if len(v) >= 3:
                    bounds = {k: v[2]}
                if len(v) >= 4:
                    units = v[3]
                if len(v) >= 5:
                    props = v[4]
                if len(v) >= 6:
                    encoding = v[5]
                res_coords.append(
                    Coordinate(
                        data=data,
                        axis=Axis(k),
                        dims=dims,
                        bounds=bounds,
                        units=units,
                        properties=props,
                        encoding=encoding,
                    )
                )
            else:
                raise TypeError(
                    "Expected types of coord values are following: [Number,"
                    " numpy.ndarray, dask.array.Array, tuple], but proided"
                    f" type was `{type(v)}`"
                )

        if crs is None:
            crs = Domain.guess_crs(coords)

        return Domain(coords=res_coords, crs=crs)


class GeodeticPoints(Domain):
    def __init__(self, latitude, longitude, vertical=None):
        latitude = np.array(latitude, dtype=np.float64, ndmin=1)
        longitude = np.array(longitude, dtype=np.float64, ndmin=1)
        if vertical != None:
            vertical = np.array(vertical, dtype=np.float64, ndmin=1)
            super().__init__(
                coords={
                    "latitude": (latitude, "points", "latitude"),
                    "longitude": (longitude, "points", "longitude"),
                    "vertical": (vertical, "points", "vertical"),
                },
                crs=GeogCS(6371229),
                domaintype=DomainType.POINTS,
            )
        else:
            super().__init__(
                coords={
                    "latitude": (latitude, "points", "latitude"),
                    "longitude": (longitude, "points", "longitude"),
                },
                crs=GeogCS(6371229),
                domaintype=DomainType.POINTS,
            )


class GeodeticGrid(Domain):
    def __init__(self, latitude, longitude, vertical=None):
        latitude = np.array(latitude, dtype=np.float64, ndmin=1)
        longitude = np.array(longitude, dtype=np.float64, ndmin=1)
        if vertical != None:
            vertical = np.array(vertical, dtype=np.float64, ndmin=1)
            super().__init__(
                coords={
                    "latitude": latitude,
                    "longitude": longitude,
                    "vertical": vertical,
                },
                crs=GeogCS(6371229),
            )
        else:
            # TODO: TO BE FIXED
            super().__init__(
                coords={
                    "latitude": (latitude, "latitude"),
                    "longitude": (longitude, "longitude"),
                },
                crs=GeogCS(6371229),
            )
