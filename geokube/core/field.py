from __future__ import annotations

from collections.abc import Sequence
import functools as ft
import json
import os
import uuid
import tempfile
import warnings
from html import escape
from itertools import chain
from numbers import Number
from typing import (
    Any,
    Callable,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import cartopy.crs as ccrs
import cartopy.feature as cartf
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow as pa
from shapely.geometry import MultiPolygon, Polygon
from shapely.vectorized import contains as sv_contains
import xarray as xr
import hvplot.xarray  # noqa
from dask import is_dask_collection
from xarray.core.options import OPTIONS

from ..utils import formatting, formatting_html, util_methods
from ..utils.decorators import geokube_logging
from ..utils.util_methods import convert_cftimes_to_numpy
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis, AxisType
from .errs import EmptyDataError
from .cell_methods import CellMethod
from .coord_system import CoordSystem, GeogCS, RegularLatLon, RotatedGeogCS
from .coordinate import Coordinate, CoordinateType
from .domain import Domain, DomainType, GeodeticPoints, GeodeticGrid
from .enums import MethodType, RegridMethod
from .unit import Unit
from .variable import Variable
from .domainmixin import DomainMixin

_CARTOPY_FEATURES = {
    "borders": cartf.BORDERS,
    "coastline": cartf.COASTLINE,
    "lakes": cartf.LAKES,
    "land": cartf.LAND,
    "ocean": cartf.OCEAN,
    "rivers": cartf.RIVERS,
    "states": cartf.STATES,
}


# pylint: disable=missing-class-docstring


class Field(Variable, DomainMixin):
    __slots__ = (
        "_name",
        "_domain",
        "_cell_methods",
        "_ancillary_data",
        "_id_pattern",
        "_mapping",
    )

    _LOG = HCubeLogger(name="Field")

    def __init__(
        self,
        data: Union[Number, np.ndarray, da.Array, xr.Variable, Variable],
        name: str,
        dims: Optional[Union[Tuple[Axis], Tuple[AxisType], Tuple[str]]] = None,
        coords: Optional[
            Union[
                Domain,
                Mapping[str, Union[Number, np.ndarray, da.Array]],
                Mapping[
                    str,
                    Tuple[
                        Tuple[str, ...], Union[Number, np.ndarray, da.Array]
                    ],
                ],
            ]
        ] = None,
        crs: Optional[CoordSystem] = None,
        units: Optional[Union[Unit, str]] = None,
        properties: Optional[Mapping[Hashable, str]] = None,
        encoding: Optional[Mapping[Hashable, str]] = None,
        cell_methods: Optional[CellMethod] = None,
        ancillary: Optional[
            Mapping[Hashable, Union[np.ndarray, Variable]]
        ] = None,
    ) -> None:
        super().__init__(
            data=data,
            units=units,
            dims=dims,
            properties=properties,
            encoding=encoding,
        )
        self._ancillary = None
        self._name = name
        self._domain = (
            coords
            if isinstance(coords, Domain)
            else Domain._make_domain_from_coords_dict_dims_and_crs(
                coords=coords, dims=dims, crs=crs
            )
        )

        self._cell_methods = cell_methods
        if ancillary is not None:
            if not isinstance(ancillary, dict):
                raise TypeError(
                    "Expected type of `ancillary` argument is dict, but the"
                    f" provided one if {type(ancillary)}"
                )
            res_anc = {}
            for k, v in ancillary.items():
                if not isinstance(v, (np.ndarray, da.Array, Variable, Number)):
                    raise TypeError(
                        "Expected type of single ancillary variable is:"
                        " `numpy.ndarray`, `dask.Array`, `geokube.Variable`,"
                        f" or `Number`, but the provided one if {type(v)}"
                    )
                # TODO: what should be axis and dims for ancillary variables? SHould it be `Variable`?
                res_anc[k] = Variable(data=v)
            self._ancillary = res_anc

    def __str__(self) -> str:
        return (
            f"Field {self.name}:{self.ncvar} with cell method:"
            f" {self.cell_methods}"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def ncvar(self) -> str:
        return self._encoding.get("name", self.name)

    @property
    def cell_methods(self) -> Optional[CellMethod]:
        return self._cell_methods

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def coords(self):
        return self._domain._coords

    @property
    def ancillary(self) -> Optional[Mapping[Hashable, Variable]]:
        return self._ancillary

    def __contains__(self, key):
        return key in self.domain

    def __getitem__(self, key):
        return self.domain[key]

    def __repr__(self) -> str:
        return self.to_xarray(encoding=False).__repr__()

    #        return formatting.array_repr(self.to_xarray())

    def _repr_html_(self):
        return self.to_xarray(encoding=False)._repr_html_()
        # if OPTIONS["display_style"] == "text":
        #     return f"<pre>{escape(repr(self.to_xarray()))}</pre>"
        # return formatting_html.array_repr(self)

    def __next__(self):
        for k, v in self.domain._coords.items():
            yield k, v
        raise StopIteration

    # geobbox and locations operates also on dependent coordinates
    # they refer only to GeoCoordinates (lat/lon)
    # TODO: Add Vertical
    @geokube_logging
    def geobbox(
        self,
        north: Number | None = None,
        south: Number | None = None,
        west: Number | None = None,
        east: Number | None = None,
        top: Number | None = None,
        bottom: Number | None = None,
    ) -> Field:
        """
        Subset a field using a bounding box.

        Subsets the original field with the given bounding box.  If a
        bound is omitted or `None`, no subsetting takes place in that
        direction.  At least one bound must be provided.

        Parameters
        ----------
        north, south, west, east : number or None, optional
            Horizontal bounds.
        top, bottom : number or None, optional
            Vertical bounds.

        Returns
        -------
        Field
            A field with the coordinate values between given bounds.

        Raises
        ------
        KeyError
            If no bound is provided.

        """
        if not util_methods.is_atleast_one_not_none(
            north, south, west, east, top, bottom
        ):
            raise KeyError(
                "At least on of the following must be defined: [north, south,"
                " west, east, top, bottom]!"
            )
        return self._geobbox_idx(
            south=south,
            north=north,
            west=west,
            east=east,
            top=top,
            bottom=bottom,
        )

    def _geobbox_cartopy(self, south, north, west, east, top, bottom):
        # TODO: add vertical also
        domain = self._domain

        ind_lat = domain.is_latitude_independent
        ind_lon = domain.is_longitude_independent
        if ind_lat and ind_lon:
            idx = {
                domain.latitude.name: np.s_[south:north]
                if util_methods.is_nondecreasing(domain.latitude.data)
                else np.s_[north:south],
                domain.longitude.name: np.s_[west:east]
                if util_methods.is_nondecreasing(domain.longitude.data)
                else np.s_[east:west],
            }
            return self.sel(indexers=idx, roll_if_needed=roll_if_needed)

        # Specifying the corner points of the bounding box in the rectangular
        # coordinate system (`cartopy.crs.PlateCarree()`).
        lats = np.array([south, south, north, north], dtype=np.float32)
        lons = np.array([west, east, west, east], dtype=np.float32)

        # Transforming the corner points of the bounding box from the
        # rectangular coordinate system (`cartopy.crs.PlateCarree`) to the
        # coordinate system of the field.
        plate = ccrs.PlateCarree()
        pts = domain.crs.as_cartopy_crs().transform_points(
            src_crs=plate, x=lons, y=lats
        )
        x, y = pts[:, 0], pts[:, 1]

        # Spatial subseting.
        idx = {
            domain[AxisType.LATITUDE].dims[1].ncvar: np.s_[x.min() : x.max()],
            domain[AxisType.LATITUDE].dims[0].ncvar: np.s_[y.min() : y.max()],
        }
        ds = (
            self._check_and_roll_longitude(self.to_xarray(), idx)
            if roll_if_needed
            else self.to_xarray()
        )
        return Field.from_xarray(
            ds=ds.sel(indexers=idx),
            ncvar=self.ncvar,
            copy=False,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )

    def _geobbox_idx(
        self,
        south: Number,
        north: Number,
        west: Number,
        east: Number,
        top: Number | None = None,
        bottom: Number | None = None,
    ):
        field = self
        lat = field.domain.get(AxisType.LATITUDE)
        lon = field.domain.get(AxisType.LONGITUDE)

#        lat, lon = field.latitude, field.longitude

        # Vertical
        # NOTE: In this implementation, vertical is always considered an
        # independent coordinate.
        if top is not None or bottom is not None:
            try:
                vert = field.vertical
            except KeyError:
                vert = None
            # TODO: Reconsider `not vert.shape`.
            if vert is None or not vert.shape:
                raise ValueError(
                    "'top' and 'bottom' must be None because there is no "
                    "vertical coordinate or it is constant"
                )
            vert_incr = util_methods.is_nondecreasing(vert.data)
            if vert.attrs.get("positive") == "down":
                top = None if top is None else -top
                bottom = None if bottom is None else -bottom
                vert_incr = ~vert_incr
            vert_slice = np.s_[bottom:top] if vert_incr else np.s_[top:bottom]
            vert_idx = {vert.name: vert_slice}
            field = field.sel(indexers=vert_idx, roll_if_needed=True)

        idx = {}
        independent = False
        if lat is not None and field.is_latitude_independent:
            # Case of latitude and longitude being independent.
            lat_incr = util_methods.is_nondecreasing(lat.data)
            lat_slice = np.s_[south:north] if lat_incr else np.s_[north:south]
            idx[lat.name] = lat_slice
            independent = True

        if lon is not None and field.is_longitude_independent:
            lon_incr = util_methods.is_nondecreasing(lon.data)
            lon_slice = np.s_[west:east] if lon_incr else np.s_[east:west]
            idx[lon.name] = lon_slice
            independent = True

        if independent:
            return field.sel(indexers=idx, roll_if_needed=True)

        if lat is not None and (not field.is_latitude_independent):
            # Case of latitude and longitude being dependent.
            # Specifying the mask(s) and extracting the indices that correspond
            # to the inside the bounding box.
            lat_mask = util_methods.is_between(lat.data, south, north)
            # TODO: Clarify why this is required.
            if lat_mask.sum() == 0:
                lat_mask = util_methods.is_between(lat.data, north, south)
        else:
            lat_mask = None

        if lon is not None and (not field.is_longitude_independent):
            lon_mask = util_methods.is_between(lon.data, west, east)
            if lon_mask.sum() == 0:
                lon_mask = util_methods.is_between(lon.data, east, west)
        else:
            lon_mask = None

        if (lat_mask is not None and lon_mask is not None):
            nonzero_idx = np.nonzero(lat_mask & lon_mask)
            idx = {
                lat.dims[i].name: np.s_[incl_idx.min() : incl_idx.max() + 1]
                for i, incl_idx in enumerate(nonzero_idx)
            }
        elif lat_mask is not None:
            nonzero_idx = np.nonzero(lat_mask)
            idx = idx.update({
                lat.dims[i].name: np.s_[incl_idx.min() : incl_idx.max() + 1]
                for i, incl_idx in enumerate(nonzero_idx)
            })
        elif lon_mask is not None:
            nonzero_idx = np.nonzero(lon_mask)
            idx = idx.update({
                lon.dims[i].name: np.s_[incl_idx.min() : incl_idx.max() + 1]
                for i, incl_idx in enumerate(nonzero_idx)
            })
        else:
            warn("'field' does not have latitude nor longitude dimensions")

        dset = field.to_xarray(encoding=False)
        dset = field._check_and_roll_longitude(dset, idx)
        dset = dset.isel(indexers=idx)

        return Field.from_xarray(
            ds=dset,
            ncvar=self.name,
            copy=False,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )

    def locations(
        self,
        latitude: Number | Sequence[Number],
        longitude: Number | Sequence[Number],
        vertical: Number | Sequence[Number] | None = None,
    ) -> (
        Field
    ):  # points are expressed as arrays for coordinates (dep or ind) lat/lon/vertical
        """
        Select points with given coordinates from a field.

        Subsets the original field by selecting only the points with
        provided coordinates and returns a new field with these points.
        Uses the nearest neighbor method.  The resulting field has a
        domain with the points nearest to the provided coordinates.

        Parameters
        ----------
        latitude, longitude : array-like or number
            Latitude and longitude coordinate values.  Must be of the
            same shape.
        vertical : array-like or number or None, optional
            Verical coordinate values.  If given and not `None`, must be
            of the same shape as `latitude` and `longitude`.

        Returns
        -------
        Field
            A field with a point domain that contains given locations.

        Examples
        --------
        >>> result = field.locations(latitude=40, longitude=35)
        >>> result.latitude.values
        array([40.86], dtype=float32)
        >>> result.longitude.values
        array([34.99963], dtype=float32)

        Vertical coordinate is optional.  If provided, the vertical axis
        of the resulting field is also expressed with points:

        >>> result = field.locations(
        ...     latitude=40,
        ...     longitude=35,
        ...     vertical=-2
        ... )
        >>> result.latitude.values
        array([40.86], dtype=float32)
        >>> result.longitude.values
        array([34.99963], dtype=float32)
        >>> result.vertical.values
        array([2.5010786], dtype=float32)

        It is possible to provide the coordinates of multiple points at
        once with an array-like object.  In that case, `latitude`,
        `longitude`, and `vertical` must have the same length.

        >>> result = temperature_field.locations(
        ...     latitude=[40, 41],
        ...     longitude=[32, 35],
        ...     vertical=[-2, -5]
        ... )
        >>> result.latitude.values
        array([40.86   , 40.99889], dtype=float32)
        >>> result.longitude.values
        array([31.99963, 34.99963], dtype=float32)
        >>> result.vertical.values
        array([2.5010786, 2.5010786], dtype=float32)
        """
        return self._locations_idx(
            latitude=latitude, longitude=longitude, vertical=vertical
        )

    def interpolate(self, domain: Domain, method: str = "nearest") -> Field:
        # TODO: Add vertical support.
        # if (
        #     {c.axis_type for c in domain.coords.values() if c.is_dimension}
        #     != {AxisType.LATITUDE, AxisType.LONGITUDE}
        # ):
        #     raise NotImplementedError(
        #         "'domain' can have only latitude and longitude at the moment"
        #     )

        dset = self.to_xarray(encoding=False)
        lat, lon = domain.latitude.values, domain.longitude.values
        if self.is_latitude_independent and self.is_longitude_independent:
            if domain.type is DomainType.POINTS:
                dim_lat = dim_lon = "points"
            else:
                dim_lat, dim_lon = self.latitude.name, self.longitude.name
            interp_coords = {
                self.latitude.name: xr.DataArray(data=lat, dims=dim_lat),
                self.longitude.name: xr.DataArray(data=lon, dims=dim_lon),
            }
            dset_interp = dset.interp(coords=interp_coords, method=method)
        else:
            if domain.type is DomainType.POINTS:
                pts = self.domain.crs.as_cartopy_crs().transform_points(
                    src_crs=domain.crs.as_cartopy_crs(), x=lon, y=lat
                )
                x, y = pts[..., 0], pts[..., 1]
                interp_coords = {
                    self.x.name: xr.DataArray(data=x, dims="points"),
                    self.y.name: xr.DataArray(data=y, dims="points"),
                }
            else:
                if lat.ndim == lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
                pts = self.domain.crs.as_cartopy_crs().transform_points(
                    src_crs=ccrs.PlateCarree(), x=lon, y=lat
                )
                x, y = pts[..., 0], pts[..., 1]
                dims = (domain.latitude.name, domain.longitude.name)
                grid = xr.Dataset(
                    data_vars={self.x.name: (dims, x), self.y.name: (dims, y)},
                    coords=domain.to_xarray(encoding=False),
                )
                dset = dset.drop(
                    labels=(self.latitude.name, self.longitude.name)
                )
                interp_coords = {
                    self.x.name: grid[self.x.name],
                    self.y.name: grid[self.y.name],
                }
            dset_interp = dset.interp(coords=interp_coords, method=method)
            dset_interp = dset_interp.drop(labels=[self.x.name, self.y.name])

        dset_interp[self.name].encoding.update(dset[self.name].encoding)

        encoding_coords = " ".join(
            coord_name
            for coord_name, coord in domain.coords.items()
            if not coord.is_dimension
        )
        # NOTE: we need to manually append scalar coordinates
        # NOTE: as they are not affected by interpolation
        encoding_coords = " ".join(
            chain(
                [encoding_coords],
                [
                    coord_name
                    for coord_name, coord in self.domain.coords.items()
                    if coord.type is CoordinateType.SCALAR
                ],
            )
        )
        if encoding_coords:
            dset_interp[self.name].encoding["coordinates"] = encoding_coords
        # TODO: Fill value should depend on the data type.
        # TODO: Add xarray fillna into Field.to_xarray.
        dset_interp[self.name].encoding["_FillValue"] = -9.0e-20

        field = Field.from_xarray(
            ds=dset_interp,
            ncvar=self.name,
            copy=False,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )

        field.domain.type = domain.type
        field.domain.crs = domain.crs
        return field

    def _locations_cartopy(self, latitude, longitude, vertical=None):
        domain = self._domain

        # Specifying the location points in the rectangular coordinate system
        # (`cartopy.crs.PlateCarree()`).
        lats = np.array(latitude, dtype=np.float32, ndmin=1)
        lons = np.array(longitude, dtype=np.float32, ndmin=1)

        ind_lat = domain.is_latitude_independent
        ind_lon = domain.is_longitude_independent
        if ind_lat and ind_lon:
            idx = {
                domain.latitude.name: lats.item() if len(lats) == 1 else lats,
                domain.longitude.name: lons.item() if len(lons) == 1 else lons,
            }
        else:
            # Transforming the location points from the rectangular coordinate
            # system (`cartopy.crs.PlateCarree`) to the coordinate system of
            # the field.
            plate = ccrs.PlateCarree()
            pts = domain.crs.as_cartopy_crs().transform_points(
                src_crs=plate, x=lons, y=lats
            )
            idx = {
                domain.x.name: xr.DataArray(data=pts[:, 0], dims="points"),
                domain.y.name: xr.DataArray(data=pts[:, 1], dims="points"),
            }

        return Field.from_xarray(
            ds=self.to_xarray(encoding=False).sel(
                indexers=idx, method="nearest"
            ),
            ncvar=self.name,
            copy=False,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )

    def _locations_idx(self, latitude, longitude, vertical=None):
        field = self
        sel_kwa = {"roll_if_needed": True, "method": "nearest"}
        lats = np.array(latitude, dtype=np.float32).reshape(-1)
        lons = np.array(longitude, dtype=np.float32).reshape(-1)

        n = lats.size
        if lons.size != n:
            raise ValueError(
                "'latitude' and 'longitude' must have the same number of items"
            )

        # Vertical
        # NOTE: In this implementation, vertical is always considered an
        # independent coordinate.
        if vertical is not None:
            verts = np.array(vertical, dtype=np.float32).reshape(-1)
            if verts.size != n:
                raise ValueError(
                    "'vertical' must have the same number of items as "
                    "'latitude' and 'longitude'"
                )
            if field.vertical.attrs.get("positive") == "down":
                verts = -verts
            verts = xr.DataArray(data=verts, dims="points")
            vert_ax = Axis(name=self.vertical.name, axistype=AxisType.VERTICAL)
            field = field.sel(indexers={vert_ax: verts}, **sel_kwa)

        # Case of latitude and longitude being independent.
        if self.is_latitude_independent and self.is_longitude_independent:
            # TODO: Check lon values conventions.
            lats = xr.DataArray(data=lats, dims="points")
            lons = xr.DataArray(data=lons, dims="points")
            idx = {self.latitude.name: lats, self.longitude.name: lons}
            result_field = field.sel(indexers=idx, **sel_kwa)
        else:
            # TODO: Check lon values conventions if possible, otherwise raise error.
            # Case of latitude and longitude being dependent on y and x.
            # Adjusting the shape of the latitude and longitude coordinates.
            # TODO: Check if these are NumPy arrays.
            # TODO: Check axes and shapes manipulation again.
            lat_data = self.latitude.values
            lat_dims = (np.s_[:],) + (np.newaxis,) * lat_data.ndim
            lat_data = lat_data[np.newaxis, :]
            lon_data = self.longitude.values
            lon_dims = (np.s_[:],) + (np.newaxis,) * lon_data.ndim
            lon_data = lon_data[np.newaxis, :]

            # Adjusting the shape of the latitude and longitude of the
            # locations.
            lats = lats[lat_dims]
            lons = lons[lon_dims]

            # Calculating the squares of the Euclidean distance.
            lat_diff = lat_data - lats
            lon_diff = lon_data - lons
            diff_sq = lat_diff * lat_diff + lon_diff * lon_diff

            # Selecting the indices that correspond to the squares of the
            # Euclidean distance.
            # TODO: Improve vectorization.
            # TODO: Consider replacing `numpy.unravel_index` with
            # `numpy.argwhere`, using the constructs like
            # `np.argwhere(diff_sq[i] == diff_sq[i].min())[0]`.
            n, *shape = diff_sq.shape
            idx_ = tuple(
                np.unravel_index(indices=diff_sq[i].argmin(), shape=shape)
                for i in range(n)
            )
            idx_ = np.array(idx_, dtype=np.int64)

            # Spatial subseting.
            idx = {
                dim.name: xr.DataArray(data=idx_[:, i], dims="points")
                for (i,), dim in np.ndenumerate(self.latitude.dims)
            }

            result_dset = field.to_xarray(encoding=False)
            result_dset = field._check_and_roll_longitude(result_dset, idx)
            result_dset = result_dset.isel(indexers=idx)
            result_field = Field.from_xarray(
                ds=result_dset,
                ncvar=self.name,
                copy=False,
                id_pattern=self._id_pattern,
                mapping=self._mapping,
            )

        result_field.domain.crs = RegularLatLon()
        result_field.domain._type = DomainType.POINTS

        return result_field

    # consider only independent coordinates
    # TODO: we should use metpy approach (user can also specify - units)
    @geokube_logging
    def sel(
        self,
        indexers: Mapping[Union[Axis, str], Any] = None,
        roll_if_needed: bool = True,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,  # TODO: check if should be always True or False in out case
        **indexers_kwargs: Any,
    ) -> "Field":
        indexers = xr.core.utils.either_dict_or_kwargs(
            indexers, indexers_kwargs, "sel"
        )
        # TODO:
        indexers = self.domain.map_indexers(indexers)
        # TODO: indexers from this place should be always of type -> Mapping[Axis, Any]
        #   ^ it can be done in Domain
        indexers = indexers.copy()
        ds = self.to_xarray(encoding=False)

        if (
            time_ind := indexers.get(Axis("time"))
        ) is not None and util_methods.is_time_combo(time_ind):
            # NOTE: time is always independent coordinate
            try:
                idx = self.domain._process_time_combo(time_ind)
            except KeyError:
                self._LOG.warn("time axis is not present in the domain.")
            else:
                if (
                    isinstance(idx["time"], np.ndarray)
                    and len(idx["time"]) == 0
                ):
                    Field._LOG.warn("empty `time` indexer")
                    raise EmptyDataError("empty `time` indexer")
                ds = ds.isel(idx, drop=drop)
                del indexers[Axis("time")]

        if roll_if_needed:
            ds = self._check_and_roll_longitude(ds, indexers)

        indexers = {
            self.domain[k].name: v
            for k, v in indexers.items()
            if k in self.domain
        }
        indexers = {
            index_key: index_value
            for index_key, index_value in indexers.items()
            if index_key in ds.xindexes
        }

        # If selection by single lat/lon, coordinate is lost as it is not stored either in da.dims nor in da.attrs["coordinates"]
        # and then selecting this location from Domain fails
        ds_dims = set(ds.dims)
        ds = ds.sel(indexers, tolerance=tolerance, method=method, drop=drop)
        lost_dims = ds_dims - set(ds.dims)
        Field._update_coordinates(ds[self.name], lost_dims)
        return Field.from_xarray(
            ds,
            ncvar=self.name,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )

    @geokube_logging
    def _check_and_roll_longitude(self, ds, indexers) -> xr.Dataset:
        # `ds` here is passed as an argument to avoid one redundent to_xarray call
        if Axis("longitude") not in indexers:
            return ds
        if (
            self.domain[Axis("longitude")].type
            is not CoordinateType.INDEPENDENT
        ):
            # TODO: implement for dependent coordinate
            raise NotImplementedError(
                "Rolling longitude is currently supported only for independent"
                " coordinate!"
            )
        first_el, last_el = (
            self.domain[Axis("longitude")].min(),
            self.domain[Axis("longitude")].max(),
        )

        if isinstance(indexers[Axis("longitude")], slice):
            start = indexers[Axis("longitude")].start
            stop = indexers[Axis("longitude")].stop
            start = 0 if start is None else start
            stop = 0 if stop is None else stop
            sel_neg_conv = (start < 0) | (stop < 0)
            sel_pos_conv = (start > 180) | (stop > 180)
        else:
            vals = np.array(indexers[Axis("longitude")], ndmin=1)
            sel_neg_conv = np.any(vals < 0)
            sel_pos_conv = np.any(vals > 180)

        dset_neg_conv = first_el < 0
        dset_pos_conv = first_el >= 0
        lng_name = self.domain[Axis("longitude")].name

        if dset_pos_conv and sel_neg_conv:
            # from [0,360] to [-180,180]
            # Attributes are lost while doing `assign_coords`. They need to be reassigned (e.q. by `update`)
            roll_value = (ds[lng_name] >= 180).sum().item()
            res = ds.assign_coords(
                {lng_name: (((ds[lng_name] + 180) % 360) - 180)}
            ).roll(**{lng_name: roll_value}, roll_coords=True)
            res[lng_name].attrs.update(ds[lng_name].attrs)
            # TODO: verify of there are some attrs that need to be updated (e.g. min/max value)
            return res
        if dset_neg_conv and sel_pos_conv:
            # from [-180,-180] to [0,360]
            roll_value = (ds[lng_name] <= 0).sum().item()
            res = (
                ds.assign_coords({lng_name: (ds[lng_name] % 360)})
                .roll(**{lng_name: -roll_value}, roll_coords=True)
                .assign_attrs(**ds[lng_name].attrs)
            )
            res[lng_name].attrs.update(ds[lng_name].attrs)
            return res
        return ds

    def to_regular(self):
        # Infering latitude and longitude steps from the x and y coordinates.
        if isinstance(self.domain.crs, RotatedGeogCS):
            lat_step = self.y.values.ptp() / (self.y.values.size - 1)
            lon_step = self.x.values.ptp() / (self.x.values.size - 1)
        else:
            raise NotImplementedError(
                f"'{type(self.domain.crs).__name__}' is not supported as a "
                "type of coordinate reference system"
            )

        # Building regular latitude-longitude coordinates.
        south = self.latitude.values.min()
        north = self.latitude.values.max()
        west = self.longitude.values.min()
        east = self.longitude.values.max()
        lat = np.arange(south, north + lat_step / 2, lat_step)
        lon = np.arange(west, east + lon_step / 2, lon_step)

        return self.interpolate(
            domain=GeodeticGrid(latitude=lat, longitude=lon), method="nearest"
        )

    def extract_polygons(self, geometry, crop=True, return_mask=False):
        # Preparing geometry.
        polygons = []
        for polygon in np.asarray(geometry).flat:
            if isinstance(polygon, Polygon):
                polygons.append(polygon)
            elif isinstance(polygon, MultiPolygon):
                polygons += list(polygon.geoms)
            else:
                raise TypeError(
                    "'geometry' must contain one or more instances of "
                    f"'Polygon' or'MultiPolygon', not {type(polygon)}"
                )
        multi_polygon = MultiPolygon(polygons=polygons)

        # Preparing masks.
        # HACK: Check against `None` is provided temporarily.
        if (
            self.domain.type is not DomainType.GRIDDED
            and self.domain.type is not None
        ):
            raise NotImplementedError(
                "'self.domain.type' must be 'DomainType.GRIDDED'"
            )
        field = (
            self if isinstance(self.domain.crs, GeogCS) else self.to_regular()
        )
        lat, lon = field.latitude.values, field.longitude.values
        coords = {"latitude": lat, "longitude": lon}
        mask = xr.DataArray(coords=coords, dims=coords.keys(), name="mask")
        lon_, lat_ = np.meshgrid(lon, lat, indexing="xy")
        mask.values = sv_contains(geometry=multi_polygon, x=lon_, y=lat_)

        # Applying mask.
        data = field.to_xarray(encoding=False).where(mask, other=np.nan)

        # Cropping.
        if crop:
            lon_min, lat_min, lon_max, lat_max = multi_polygon.bounds
            lat_order = 1 if lat[0] <= lat[-1] else -1
            lon_order = 1 if lon[0] <= lon[-1] else -1
            idx = {
                field.latitude.name: np.s_[lat_min:lat_max:lat_order],
                field.longitude.name: np.s_[lon_min:lon_max:lon_order],
            }
            data = data.sel(indexers=idx)

        # Converting back to field.
        result = Field.from_xarray(data, ncvar=field.name)

        return (result, mask) if return_mask else result

    # TO CHECK
    @geokube_logging
    def regrid(
        self,
        target: Union[Domain, "Field"],
        method: Union[str, RegridMethod] = "bilinear",
        weights_path: Optional[str] = None,
        reuse_weights: bool = True,
    ) -> "Field":
        """
        Regridds present coordinate system.
        Parameters
        ----------
        target_domain : geokube.Domain or geokube.Field
            Domain which is supposed to be the result of regridding.
        method : str
            A method to use for regridding. Default: `bilinear`.
        weights_path : str, optional
            The path of the file where the interpolation weights are
            stored. Default: `None`.
        reuse_weights : bool, optional
            Whether to reuse already calculated weights or not. Default:
            `True`.
        Returns
        ----------
        field : Field
           The field with values modified by regridding query.
        Examples:
        ----------
        >>> result = field.regrid(
        ...     target_domain=target_domain,
        ...     method='bilinear'
        ... )

        """
        import xesmf as xe
        if not isinstance(target, Domain):
            if isinstance(target, Field):
                target = target.domain
            else:
                raise TypeError(
                    "'target' must be an instance of Domain or Field"
                )

        if not isinstance(method, RegridMethod):
            method = RegridMethod[str(method).upper()]

        if reuse_weights and (
            weights_path is None or not os.path.exists(weights_path)
        ):
            Field._LOG.warn("`weights_path` is None or file does not exist!")
            Field._LOG.info("`reuse_weights` turned off")
            reuse_weights = False

        names_in = {self.latitude.name: "lat", self.longitude.name: "lon"}
        names_out = {target.latitude.name: "lat", target.longitude.name: "lon"}
        coords_in = coords_out = None

        if method in {
            RegridMethod.CONSERVATIVE,
            RegridMethod.CONSERVATIVE_NORMED,
        }:
            self.domain.compute_bounds()
            lat_b_name = next(iter(self.latitude.bounds))
            lat_b = next(iter(self.latitude.bounds.values())).values
            lat_b = Domain.convert_bounds_2d_to_1d(lat_b)
            lon_b_name = next(iter(self.longitude.bounds))
            lon_b = next(iter(self.longitude.bounds.values())).values
            lon_b = Domain.convert_bounds_2d_to_1d(lon_b)
            names_in.update({lat_b_name: "lat_b", lon_b_name: "lon_b"})
            coords_in = {"lat_b": lat_b, "lon_b": lon_b}

            target.compute_bounds()
            lat_b_name = next(iter(target.latitude.bounds))
            lat_b = next(iter(target.latitude.bounds.values())).values
            lat_b = Domain.convert_bounds_2d_to_1d(lat_b)
            lon_b_name = next(iter(target.longitude.bounds))
            lon_b = next(iter(target.longitude.bounds.values())).values
            lon_b = Domain.convert_bounds_2d_to_1d(lon_b)
            names_out.update({lat_b_name: "lat_b", lon_b_name: "lon_b"})
            coords_out = {"lat_b": lat_b, "lon_b": lon_b}

        # Regridding
        in_ = self.to_xarray(encoding=False).rename(names_in)
        if coords_in:
            in_ = in_.assign_coords(coords=coords_in)
        out = target.to_xarray(encoding=False).to_dataset().rename(names_out)
        if coords_out:
            out = out.assign_coords(coords=coords_out)
        regrid_kwa = {
            "ds_in": in_,
            "ds_out": out,
            "method": method.value,
            "unmapped_to_nan": True,
            "filename": weights_path,
        }
        try:
            regridder = xe.Regridder(**regrid_kwa, reuse_weights=reuse_weights)
        except PermissionError:
            regridder = xe.Regridder(**regrid_kwa)
        result = regridder(in_, keep_attrs=True, skipna=False)
        result = result.rename({v: k for k, v in names_in.items()})
        result[self.name].encoding = in_[self.name].encoding
        if not isinstance(target.crs, GeogCS):
            missing_coords = {
                coord.name: out.coords[coord.name]
                for coord in (target.x, target.y)
                if coord.name not in result.coords
            }
            result = result.assign_coords(coords=missing_coords)
            result[self.name].encoding["coordinates"] = "latitude longitude"
        # After regridding those attributes are not valid!
        util_methods.clear_attributes(result, attrs="cell_measures")
        field_out = Field.from_xarray(
            ds=result,
            ncvar=self.name,
            copy=False,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )
        field_out.domain._crs = target.crs
        field_out.domain._type = target.type
        return field_out

    # TO CHECK
    @geokube_logging
    def resample(
        self,
        operator: Union[Callable, MethodType, str],
        frequency: str,
        **resample_kwargs,
    ) -> "Field":
        """
        Perform resampling along the available `time` coordinate.
        Adjust appropriately time bounds.

        Parameters
        ----------
        operator : callable or str
            Callable-object used for aggregation or string
            representation of a function.  Currently supported are the
            methods of ``geokube.MethodType``.
        frequency :  str
            Expected resampling frequency.

        Returns
        ----------
        field : Field
            The field with values after resampling procedure.

        Examples:
        ----------
        Resample to day frequency taking the maximum over the elements
        in each day:
        >>> res = field.resample("maximum", frequency='1D')

        Resample to two-monts frequency taking the sum over each two
        months:
        >>> resulting_field = field.resample("sum", frequency='2M')

        """
        if callable(operator):
            func = operator
        else:
            if isinstance(operator, str):
                operator_func = MethodType(operator)
            elif isinstance(operator, MethodType):
                operator_func = operator
            else:
                raise TypeError(
                    "Operator must be `str`, `MethodType`, or `callable`."
                    " Provided `{type(operator)}`"
                )
            if operator_func is MethodType.UNDEFINED:
                methods = {
                    method.value[0]
                    for method in MethodType.__members__.values()
                }
                methods.discard("<undefined>")
                raise ValueError(
                    f"Provided operator '{operator}' was not found! Available"
                    f" operators are: {sorted(methods)}!"
                )
            func = (
                operator_func.dask_operator
                if is_dask_collection(self)
                else operator_func.numpy_operator
            )

        # ################## Temporary solution for time bounds adjustmnent ######################

        # TODO: handle `formula_terms` bounds attribute
        # http://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries
        ds = self.to_xarray(encoding=False)
        if (time_bnds := self.time.bounds) is None:
            bnds_name, bnds = f"{self.time.name}_bnds", None
        else:
            ((bnds_name, bnds),) = time_bnds.items()
        if self.cell_methods and bnds is not None:
            # `closed=right` set by default for {"M", "A", "Q", "BM", "BA", "BQ", "W"} resampling codes ("D" not included!)
            # https://github.com/pandas-dev/pandas/blob/7c48ff4409c622c582c56a5702373f726de08e96/pandas/core/resample.py#L1383
            resample_kwargs.update({"closed": "right"})
            da = ds.resample(
                indexer={self.time.name: frequency}, **resample_kwargs
            )
            new_bnds = np.empty(
                shape=(len(da.groups), 2), dtype=np.dtype("datetime64[ns]")
            )
            for i, v in enumerate(da.groups.values()):
                new_bnds[i] = [bnds.values[v].min(), bnds.values[v].max()]
            # TODO: Check if this is redundant
            if bnds is None:
                Field._LOG.warn(
                    "Time bounds not defined for the cell methods!"
                )
                warnings.warn("Time bounds not defined for the cell methods!")
        else:
            da = ds.resample(
                indexer={self.time.name: frequency}, **resample_kwargs
            )
            new_bnds = np.empty(
                shape=(len(da.groups), 2), dtype=np.dtype("datetime64[ns]")
            )
            for i, v in enumerate(da.groups.values()):
                new_bnds[i] = [
                    self.time.values[v].min(),
                    self.time.values[v].max(),
                ]

        # NOTE: `reduce` spans result for all intermediate values
        # if 1st and 3rd day os selected and 1D freq is used
        # then `reduce` results in time axis for 1st, 2nd, and 3rd
        # passing for missing dates `NaN`s
        da = da.reduce(func=func, dim=self.time.name, keep_attrs=True).dropna(
            dim=self.time.name, how="all"
        )
        # NOTE: `reduce` removes all `encoding` properties
        da[self.name].encoding = ds[self.name].encoding
        res = xr.Dataset(
            da,
            coords={f"{bnds_name}": ((self.time.name, "bnds"), new_bnds)},
        )
        field = Field.from_xarray(res, ncvar=self.name)
        field.time.bounds = new_bnds
        field.domain.crs = self.domain.crs
        field.domain._type = self.domain._type

        # TODO: adjust cell_methods after resampling!

        # #########################################################################################
        return field

    @geokube_logging
    def average(self, dim: str | None = None) -> Field:
        dset = self.to_xarray(encoding=False)
        if dim is None:
            # return dset[self._name].mean().data
            result = dset[self._name].mean().data
            result[self._name].encoding = dset[self._name].encoding
            return Field.from_xarray(
                ds=xr.DataArray(data=result).to_dataset(name=self._name),
                ncvar=self.name,
                id_pattern=self._id_pattern,
                mapping=self._mapping,
                copy=False,
            )
        if self.coords[dim].is_dim:
            result_dset = dset.mean(dim=dim)
            result_dset[self._name].encoding = dset[self._name].encoding
            result_field = Field.from_xarray(
                ds=result_dset,
                ncvar=self.name,
                copy=False,
                id_pattern=self._id_pattern,
                mapping=self._mapping,
            )
            result_field.domain.crs = self._domain.crs
            result_field.domain._type = self._domain._type
            return result_field
        raise ValueError(f"'dim' {dim} is not supported for averaging")

    @geokube_logging
    def to_netcdf(self, path):
        self.to_xarray().to_netcdf(path=path)

    # TO CHECK
    @geokube_logging
    def plot(
        self,
        features=None,
        gridlines=None,
        gridline_labels=None,
        subplot_kwargs=None,
        projection=None,
        figsize=None,
        robust=None,
        aspect=None,
        save_path=None,
        save_kwargs=None,
        clean_image=False,
        **kwargs,
    ):
        axis_names = self.domain._axis_to_name
        time = self.coords.get(axis_names.get(AxisType.TIME))
        vert = self.coords.get(axis_names.get(AxisType.VERTICAL))
        lat = self.coords.get(axis_names.get(AxisType.LATITUDE))
        lon = self.coords.get(axis_names.get(AxisType.LONGITUDE))

        # NOTE: The argument `save_kwargs` passes the keyword arguments to the
        # `savefig` method, see:
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

        # Resolving time series and layers because they do not require most of
        # processing other plot types do:
        if self._domain._type is DomainType.POINTS:
            if figsize is not None:
                kwargs["figsize"] = figsize
            n_pts = lat.size
            n_time = time.size if (time is not None and time.is_dim) else 0
            n_vert = vert.size if (vert is not None and vert.is_dim) else 0
            with np.nditer((lat.values, lon.values)) as it:
                # points = [
                #     f"{lat.name}={lat_.item():.2f} {lat.units}, "
                #     f"{lon.name}={lon_.item():.2f} {lon.units}"
                #     for lat_, lon_ in it
                # ]
                points = [
                    f"{lat_.item():.2f}°, {lon_.item():.2f}°"
                    for lat_, lon_ in it
                ]
            if aspect is None:
                # Integers determine the priority in the case of equal sizes:
                # greater number means higher priority.
                aspect = max(
                    (n_time, 1, "time_series"),
                    (n_vert, 2, "profile"),
                    (n_pts, 3, "points"),
                )[2]
            if aspect not in {"time_series", "profile", "points"}:
                raise ValueError(
                    "'aspect' must be 'time_series', 'profile', 'points', or"
                    " None"
                )
            if aspect == "time_series":
                data = self.to_xarray(encoding=False)[self.name]
                data = data.assign_coords(points=points)
                kwargs["x"] = time.name
                if n_vert > 1 or (vert is not None and vert.name in data.dims):
                    kwargs.setdefault("row", vert.name)
                if "crs" in data.coords:
                    data = data.drop("crs")
                plot = data.plot.line(**kwargs)
                if "row" not in kwargs and "col" not in kwargs:
                    for line in plot:
                        line.axes.set_title("Point Time Series")
                if clean_image:
                    plot.axes.set_axis_off()
                    plot.axes.set_title('')
                if save_path:
                    fig = plot[0].figure
                    fig.tight_layout()
                    fig.savefig(save_path, **(save_kwargs or {}))
                return plot
            if aspect == "profile":
                data = self.to_xarray(encoding=False)[self.name]
                data = data.assign_coords(points=points)
                if vert.attrs.get("positive") == "down":
                    data = data.reindex(
                        indexers={vert.name: data.coords[vert.name][::-1]},
                        copy=False,
                    )
                    data.coords[vert.name] = -data.coords[vert.name]
                    # vert.values = -vert.values[::-1]
                kwargs["y"] = vert.name
                if n_time > 1 or (time is not None and time.name in data.dims):
                    kwargs.setdefault("col", time.name)
                if "crs" in data.coords:
                    data = data.drop("crs")
                plot = data.plot.line(**kwargs)
                if "row" not in kwargs and "col" not in kwargs:
                    for line in plot:
                        line.axes.set_title("Point Layers")
                if clean_image:
                    plot.axes.set_axis_off()
                    plot.axes.set_title('')
                if save_path:
                    fig = plot[0].figure
                    fig.tight_layout()
                    fig.savefig(save_path, **(save_kwargs or {}))
                return plot

        # Resolving Cartopy features and gridlines:
        if features:
            features = [_CARTOPY_FEATURES[feature] for feature in features]
            if gridlines is None:
                Field._LOG.info("`gridline` turned on")
                gridlines = True
        if gridline_labels is None:
            Field._LOG.info("`gridline_labels` turned off")
            gridline_labels = False
        has_cartopy_items = bool(features or gridlines)

        # Resolving dimensions, coordinates, and coordinate system:
        dims = set()
        if time is not None:
            dims.add(time.name)
        if vert is not None:
            dims.add(vert.name)
        if lat is not None:
            dims.add(lat.name)
            if lat.is_dim:
                kwargs.setdefault("y", lat.name)
        if lon is not None:
            dims.add(lon.name)
            if lon.is_dim:
                kwargs.setdefault("x", lon.name)
        crs = self._domain.crs
        try:
            transform = (
                crs.as_cartopy_projection() if crs is not None else None
            )
        except NotImplementedError:
            # HACK: This is used in the cases where obtaining Cartopy
            # projections is not implemented.
            transform = None
            kwargs.setdefault("x", lon.name)
            kwargs.setdefault("y", lat.name)
        plate = ccrs.PlateCarree

        if len(dims) in {3, 4}:
            if time is not None and time.name in dims and time.size > 1:
                kwargs.setdefault("col", time.name)
            if vert is not None and vert.name in dims and vert.size > 1:
                kwargs.setdefault("row", vert.name)

        # Resolving subplot keyword arguments including `projection`:
        subplot_kwa = {} if subplot_kwargs is None else {**subplot_kwargs}
        if projection is None:
            if has_cartopy_items:
                subplot_kwa["projection"] = projection = plate()
                if transform is None:
                    transform = plate()
            elif isinstance(transform, plate):
                transform = None
            if transform is not None:
                has_cartopy_items = True
                subplot_kwa["projection"] = projection = plate()
        else:
            has_cartopy_items = True
            if isinstance(projection, CoordSystem):
                projection = projection.as_cartopy_projection()
            subplot_kwa["projection"] = projection
            if transform is None:
                transform = plate()
        if subplot_kwa:
            kwargs["subplot_kws"] = subplot_kwa

        # Resolving other keyword arguments including `transform`, `figsize`,
        # and `robust`:
        kwa = {"transform": transform, "figsize": figsize, "robust": robust}
        for name, arg in kwa.items():
            if arg is not None:
                kwargs[name] = arg

        # Creating plot:
        dset = self.to_xarray(encoding=False)
        if "crs" in dset.coords:
            dset = dset.drop("crs")
        # HACK: This should be only:
        # `if self._domain._type is DomainType.GRIDDED:`
        # Checking against `None` is provided temporary for testing.
        if (
            self._domain._type is DomainType.GRIDDED
            or self._domain._type is None
        ):
            data = dset[self.name]
            plot = data.plot(**kwargs)
        elif self._domain._type is DomainType.POINTS:
            data = xr.Dataset(
                data_vars={
                    self.name: dset[self.name],
                    "lat": dset.coords["latitude"],
                    "lon": dset.coords["longitude"],
                }
            )
            kwargs.update(
                {"x": "lon", "y": "lat", "hue": self.name, "zorder": np.inf}
            )
            plot = data.plot.scatter(**kwargs)
        else:
            raise NotImplementedError(
                "'domain.type' of must be 'DomainType.GRIDDED' or "
                "'DomainType.POINTS'"
            )

        # Adding and modifying axis elements:
        # axes = np.array(getattr(plot, 'axes', plot), copy=False, ndmin=1)

        # Adding gridlines and Cartopy features (borders, coastline, lakes,
        # land, ocean, rivers, or states) to all plot axes:
        if has_cartopy_items:
            axes = np.array(plot.axes, copy=False, ndmin=1)
            if features:
                for ax in axes.flat:
                    for feature in features:
                        ax.add_feature(feature)
            if gridlines:
                for ax in axes.flat:
                    ax.gridlines(draw_labels=gridline_labels)

            # NOTE: This is a fix that enables using axis labels and units from
            # the domain, as well as plotting axes labels and ticks when
            # Cartopy transform, projection, or features are used. See:
            # https://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels
            if (
                (projection is None or isinstance(projection, plate))
                and lat is not None
                and lon is not None
                and not gridline_labels
            ):
                coords = data.coords

                lat_coord = coords[lat.name]
                lat_attrs = lat_coord.attrs
                lat_name = (
                    lat_attrs.get("long_name")
                    or lat_attrs.get("standard_name")
                    or lat_coord.name
                    or "latitude"
                )
                if (lat_units := lat.units) and lat_units != "dimensionless":
                    lat_name = f"{lat_name} [{lat_units}]"
                lat_values = lat.values
                lat_min, lat_max = lat_values.min(), lat_values.max()

                lon_coord = coords[lon.name]
                lon_attrs = lon_coord.attrs
                lon_name = (
                    lon_attrs.get("long_name")
                    or lon_attrs.get("standard_name")
                    or lon_coord.name
                    or "longitude"
                )
                if (lon_units := lon.units) and lon_units != "dimensionless":
                    lon_name = f"{lon_name} [{lon_units}]"
                lon_values = lon.values
                lon_min, lon_max = lon_values.min(), lon_values.max()

                ax = axes.item(0)
                x_ticks = ax.get_xticks()
                x_ticks = x_ticks[(x_ticks >= lon_min) & (x_ticks <= lon_max)]
                y_ticks = ax.get_yticks()
                y_ticks = y_ticks[(y_ticks >= lat_min) & (y_ticks <= lat_max)]

                if axes.ndim == 2:
                    for ax in axes[-1, :].flat:
                        ax.set_xlabel(lon_name)
                        ax.set_xticks(x_ticks)
                    for ax in axes[:, 0].flat:
                        ax.set_ylabel(lat_name)
                        ax.set_yticks(y_ticks)
                else:
                    for ax in axes.flat:
                        ax.set_xlabel(lon_name)
                        ax.set_ylabel(lat_name)
                        ax.set_xticks(x_ticks)
                        ax.set_yticks(y_ticks)

        if clean_image:
            if isinstance(plot.axes, np.ndarray):
                for axis in plot.axes.flat:
                    axis.set_axis_off()
                    axis.set_title('')
            else:
                plot.axes.set_axis_off()
                plot.axes.set_title('')
        if save_path:
            if hasattr(plot, 'fig'):
                fig = plot.fig
            elif hasattr(plot, 'figure'):
                fig = plot.figure
            else:
                raise NotImplementedError()
            fig.tight_layout()
            # fig.tight_layout(pad=0, h_pad=0, w_pad=0)
            fig.savefig(save_path, **(save_kwargs or {}))

        return plot

    def to_image(
        self,
        filepath,
        width,
        height,
        dpi=100,
        format='png',
        transparent=True,
        bgcolor='FFFFFF'
    ):
        # NOTE: This method assumes default DPI value.
        f = self
        if self.domain.crs != GeogCS(6371229):
            f = self.to_regular()
#        dpi = plt.rcParams['figure.dpi']
        w, h = width / dpi, height / dpi
        f.plot(
            figsize=(w, h),
            add_colorbar=False,
            save_path=filepath,
            save_kwargs={
                'transparent': transparent,
                'pad_inches': 0,
                'dpi': dpi,
                # 'bbox_inches': [[0, 0], [w, h]]
            },
            clean_image=True
        )

    def to_geojson(self, target=None):
        self.load()
        if self.domain.type is DomainType.POINTS:
            if self.latitude.size != 1 or self.longitude.size != 1:
                raise NotImplementedError(
                    "'self.domain' must have exactly 1 point"
                )
            coords = [self.longitude.item(), self.latitude.item()]
            result = {
                "type": "FeatureCollection",
                "units": {self.name: str(self.units)},
                "features": [],
            }
            for time in self.time.values.flat:
                time_ = pd.to_datetime(time).strftime("%Y-%m-%dT%H:%M")
                value = self.sel(time=time_) if self.time.size > 1 else self
                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coords},
                    "properties": {"time": time_, self.name: float(value)},
                }
                result["features"].append(feature)
        elif (
            self.domain.type is DomainType.GRIDDED or self.domain.type is None
        ):
            # HACK: The case `self.domain.type is None` is included to be able
            # to handle undefined domain types temporarily.
            if self.time.size == 1:
                result = {}
            else:
                raise NotImplementedError(
                    f"multiple times are not supported for geojson"
                )
            field = (
                self
                if isinstance(self.domain.crs, GeogCS)
                else self.to_regular()
            )
            axis_names = field.domain._axis_to_name
            lon_min = self.longitude.min().item()
            lat_min = self.latitude.min().item()
            lon_max = self.longitude.max().item()
            lat_max = self.latitude.max().item()              
            grid_x, grid_y = field.domain._infer_resolution()/2.0
            for time in self.time.values.flat:
                time_ = pd.to_datetime(time).strftime("%Y-%m-%dT%H:%M")
                time_data = {
                    "type": "FeatureCollection",
                    "date": time_,
                    "bbox": [
                        lon_min,  # West
                        lat_min,  # South
                        lon_max,  # East
                        lat_max,  # North
                    ],
                    "units": {self.name: str(self.units)},
                    "features": [],
                }
                for lat in field.latitude.values.flat:
                    for lon in field.longitude.values.flat:
                        idx = {
                            axis_names[AxisType.LATITUDE]: lat,
                            axis_names[AxisType.LONGITUDE]: lon,
                        }
                        # if self.time.shape:
                        if self.time.size > 1:
                            idx[axis_names[AxisType.TIME]] = time_
                        # TODO: Check whether this works now:
                        # this gives an error if only 1 time is selected before to_geojson()
                        # 
                        # Polygon:
                        # for each lat/lon we have to define a polygon with lan/lon centered
                        # the cell length depends on the grid resolution (that should be computed)
                        value = field.sel(**idx)
                        lonv = lon.item()
                        latv = lat.item()
                        lon_lower = np.clip(lonv - grid_x, amin=lon_min)
                        lat_upper = np.clip(latv + grid_y, amax=lat_max)
                        lon_upper = np.clip(lonv + grid_x, amax=lon_max)
                        lat_lower = np.clip(latv - grid_y, amin=lat_min)                        
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates":[  
                                   [ [lon_lower, lat_upper], [lon_upper, lat_upper],
                                     [lon_upper, lat_lower], [lon_lower, lat_lower],
                                     [lon_lower, lat_upper]
                                   ]
                                ]
                            },
                            "properties": {self.name: float(value)},
                        }
                        time_data["features"].append(feature)
                result = time_data
        else:
            raise NotImplementedError(
                f"'self.domain.type' is {self.domain.type}, which is currently"
                " not supported"
            )

        if target is not None:
            with open(target, mode="w") as file:
                json.dump(result, file, indent=4)

        return result

    @geokube_logging
    def hvplot(self, aspect=None, boxplot=False, **kwargs):
        # NOTE: See https://hvplot.holoviz.org/user_guide/Customization.html
        # for the details on what can be passed with `kwargs`.

        axis_names = self.domain._axis_to_name
        time = self.coords.get(axis_names.get(AxisType.TIME))
        vert = self.coords.get(axis_names.get(AxisType.VERTICAL))
        lat = self.coords.get(axis_names.get(AxisType.LATITUDE))
        lon = self.coords.get(axis_names.get(AxisType.LONGITUDE))

        kwargs.setdefault("widget_location", "bottom")

        dset = self.to_xarray(encoding=False)
        if "crs" in dset.coords:
            dset = dset.drop("crs")
        if (
            vert is not None
            and vert.is_dim
            and vert.attrs.get("positive") == "down"
        ):
            dset = dset.reindex(
                indexers={vert.name: dset.coords[vert.name][::-1]},
                copy=False,
            )
            dset.coords[vert.name] = -dset.coords[vert.name]

        # Considering the case when boxplot is required.
        if boxplot:
            group = kwargs.get("groupby")
            if group == "vertical":
                kwargs["groupby"] = vert.name
            elif group and not isinstance(group, str) and "vertical" in group:
                group = list(group)
                idx = group.index("vertical")
                group[idx] = vert.name
                kwargs["groupby"] = group

            data = dset[self.name]

            if self._domain._type is DomainType.POINTS:
                with np.nditer((lat.values, lon.values)) as it:
                    points = [
                        f"{lat_.item():.2f}°, {lon_.item():.2f}°"
                        for lat_, lon_ in it
                    ]
                data = data.assign_coords(points=points)

            return data.hvplot.box(y=self.name, **kwargs)

        # Working with `DomainType.POINTS`.
        if self._domain._type is DomainType.POINTS:
            n_pts = lat.size
            n_time = time.size if (time is not None and time.is_dim) else 0
            n_vert = vert.size if (vert is not None and vert.is_dim) else 0
            if vert is None or vert.is_dim:
                # NOTE: Case when vertical is given as a profile.
                vals = (lat.values, lon.values)
                with np.nditer(vals) as it:
                    # NOTE: This approach might result in an error when the
                    # string representations of multiple points are equal.
                    # points = [
                    #     f'{lat.name}={lat_.item():.2f} {lat.units}, '
                    #     f'{lon.name}={lon_.item():.2f} {lon.units}'
                    #     for lat_, lon_ in it
                    # ]
                    points = [
                        f"{lat_.item():.2f}°, {lon_.item():.2f}°"
                        for lat_, lon_ in it
                    ]
                if aspect is None:
                    # Integers determine the priority in the case of equal:
                    # sizes greater number means higher priority.
                    aspect = max(
                        (n_time, 1, "time_series"),
                        (n_vert, 2, "profile"),
                        (n_pts, 3, "points"),
                    )[2]
            else:
                # NOTE: Case when vertical is given as points.
                # NOTE: This approach might result in an error when the
                # string representations of multiple points are equal.
                dset.coords[vert.name] = -dset.coords[vert.name]
                vals = (lat.values, lon.values, dset.coords[vert.name].values)
                with np.nditer(vals) as it:
                    points = [
                        f"{lat_.item():.2f}°, {lon_.item():.2f}° "
                        f"{vert_.item():.2f} {vert.units}"
                        for lat_, lon_, vert_ in it
                    ]
                if aspect is None:
                    # Integers determine the priority in the case of equal:
                    # sizes greater number means higher priority.
                    aspect = max(
                        (n_time, 1, "time_series"),
                        (n_pts, 2, "points"),
                    )[2]
            if aspect == "time_series":
                data = dset[self.name]
                data = data.assign_coords(points=points)
                kwargs.update({"geo": False, "tiles": False})
                return data.hvplot(x=time.name, by="points", **kwargs)
            if aspect == "profile":
                data = dset[self.name]
                data = data.assign_coords(points=points)
                kwargs.update({"geo": False, "tiles": False})
                return data.hvplot(y=vert.name, by="points", **kwargs)
            if aspect == "points":
                data_vars = {
                    self.name: dset[self.name],
                    "lat": dset.coords["latitude"],
                    "lon": dset.coords["longitude"],
                }
                if (vert is not None) and (not vert.is_dim):
                    data_vars["vert"] = dset.coords[vert.name]
                    if "groupby" not in kwargs:
                        kwargs["groupby"] = sorted(
                            self.coords.keys() - {"latitude", "longitude"}
                        )
                data = xr.Dataset(data_vars=data_vars)
                return data.hvplot.scatter(
                    x="lon",
                    y="lat",
                    c=self.name,
                    cmap=kwargs.pop("cmap", "coolwarm"),
                    colorbar=kwargs.pop("colorbar", True),
                    **kwargs,
                )
            raise ValueError(
                "'aspect' must be 'time_series', 'profile', 'points', or None"
            )

        # Working with `DomainType.GRIDDED`.
        # HACK: This should be only:
        # `if self._domain._type is DomainType.GRIDDED:`
        # Checking against `None` is provided temporary for testing.
        if (
            self._domain._type is DomainType.GRIDDED
            or self._domain._type is None
        ):
            crs = self._domain.crs
            if crs is not None:
                try:
                    crs = crs.as_cartopy_projection()
                except NotImplementedError:
                    # HACK: This is used in the cases where obtaining Cartopy
                    # projections is not implemented.
                    crs = None
                    kwargs.setdefault("x", lon.name)
                    kwargs.setdefault("y", lat.name)

            proj = kwargs.get("projection")
            if isinstance(proj, CoordSystem):
                kwargs["projection"] = proj = proj.as_cartopy_projection()

            if aspect is not None:
                if aspect == "time_series":
                    kwargs.update({"x": time.name, "y": self.name})
                    kwargs.update({"geo": False, "tiles": False})
                elif aspect == "profile":
                    kwargs.update({"x": self.name, "y": vert.name})
                    kwargs.update({"geo": False, "tiles": False})
                if crs is not None and not isinstance(crs, ccrs.PlateCarree):
                    dset = self.to_regular().to_xarray(encoding=False)

            plot_call = dset[self.name].hvplot

            if lat is not None and lat.is_dim:
                kwargs.setdefault("y", lat.name)
            if lon is not None and lon.is_dim:
                kwargs.setdefault("x", lon.name)

            if (
                crs is None
                and lat is not None
                and lon is not None
                and (lat.size > 1 or lon.size > 1)
                and aspect is None
            ):
                plot_call = plot_call.quadmesh
                kwargs.setdefault("rasterize", True)
                kwargs.setdefault("project", True)

            if (
                not (
                    (proj is None or isinstance(proj, ccrs.PlateCarree))
                    and (crs is None or isinstance(crs, ccrs.PlateCarree))
                )
                and "x" not in kwargs
                and "y" not in kwargs
                and lat is not None
                and lon is not None
            ):
                plot_call = plot_call.quadmesh
                kwargs["crs"] = crs
                kwargs.setdefault("rasterize", True)
                kwargs.setdefault("project", True)
                lat_name = lat.attrs.get("long_name", lat.name)
                if (lat_units := lat.attrs.get("units")) is not None:
                    lat_name = f"{lat_name} ({lat_units})"
                lon_name = lon.attrs.get("long_name", lon.name)
                if (lon_units := lon.attrs.get("units")) is not None:
                    lon_name = f"{lon_name} ({lon_units})"
                kwargs.update({"xlabel": lon_name, "ylabel": lat_name})

            # TODO: Consider improving the logic (handling conditions).
            if (
                kwargs.get("tiles")
                and lat is not None
                and lon is not None
                and kwargs.get("x") == lon.name
                and kwargs.get("y") == lat.name
                and aspect is None
            ):
                lat_name = lat.attrs.get("long_name", lat.name)
                if (lat_units := lat.attrs.get("units")) is not None:
                    lat_name = f"{lat_name} ({lat_units})"
                lon_name = lon.attrs.get("long_name", lon.name)
                if (lon_units := lon.attrs.get("units")) is not None:
                    lon_name = f"{lon_name} ({lon_units})"
                kwargs.update({"xlabel": lon_name, "ylabel": lat_name})

            return plot_call(**kwargs)

        raise NotImplementedError(
            "'domain.type' must be 'DomainType.GRIDDED' or 'DomainType.POINTS'"
        )

    @geokube_logging
    def box_plot(self, by=None, orientation="vertical", **kwargs):
        # NOTE: `kwargs` are passed directly or in a slightly modified form
        # to `plotly.express.box`. For more details, see the official
        # documentation:
        # * https://plotly.github.io/plotly.py-docs/generated/plotly.express.box.html
        # * https://plotly.com/python/box-plots/

        axis_names = self.domain._axis_to_name
        time = self.coords.get(axis_names.get(AxisType.TIME))
        vert = self.coords.get(axis_names.get(AxisType.VERTICAL))
        lat = self.coords.get(axis_names.get(AxisType.LATITUDE))
        lon = self.coords.get(axis_names.get(AxisType.LONGITUDE))

        if by is None:
            by_ = None
        elif by == "points" and self._domain._type is DomainType.POINTS:
            by_ = "points"
        else:
            if by in (dim_types := {ax.value[0] for ax in axis_names.keys()}):
                by_ = getattr(self, by).name
            elif by not in (dim_names := {*axis_names.values()}):
                raise ValueError(
                    "'by' must be 'None' or one of the following: "
                    f"{sorted(dim_types | dim_names)}"
                )

        if orientation in {"h", "horizontal"}:
            kwargs.setdefault("x", self.name)
            if by_:
                kwargs.setdefault("y", by_)
        elif orientation in {"v", "vertical"}:
            kwargs.setdefault("y", self.name)
            if by_:
                kwargs.setdefault("x", by_)
        else:
            raise ValueError(
                "'orientation' must be either 'horizontal' ('h') or "
                "'vertical' ('v')"
            )

        dset = self.to_xarray(encoding=False)
        if "crs" in dset.coords:
            dset = dset.drop("crs")
        if (
            vert is not None
            and vert.is_dim
            and vert.attrs.get("positive") == "down"
        ):
            dset = dset.reindex(
                indexers={vert.name: dset.coords[vert.name][::-1]},
                copy=False,
            )
            dset.coords[vert.name] = -dset.coords[vert.name]

        darr = dset[self.name]
        if self._domain._type is DomainType.POINTS:
            with np.nditer((lat.values, lon.values)) as it:
                points = [
                    f"{lat_.item():.2f}°, {lon_.item():.2f}°"
                    for lat_, lon_ in it
                ]
                darr = darr.assign_coords(points=points)

        df = darr.to_dataframe()
        df_ = df.index.to_frame()
        df_.index = np.arange(df_.shape[0])
        for col_name in df.columns.to_numpy().flat:
            df_[col_name] = df[col_name].to_numpy()
        af = kwargs.get("animation_frame")
        if af and af == time.name:
            kwargs["animation_frame"] = animation_name = af + "_"
            df_[animation_name] = df_[af].astype(str)

        fig = px.box(df_, **kwargs)

        if af:
            min_, max_ = np.nanmin(self.values), np.nanmax(self.values)
            margin = 0.05 * (np.nanmax(self.values) - np.nanmin(self.values))
            bounds = [min_ - margin, max_ + margin]
            name = f"{'x' if orientation[0] == 'h' else 'y'}axis_range"
            kwa = {name: bounds}
            fig.update_layout(**kwa)
            # for f in fig.frames:
            #     f.layout.update(**kwa)

        return fig

    @geokube_logging
    def to_xarray(self, encoding=True) -> xr.Dataset:
        data_vars = {}
        var_name = self.ncvar if encoding else self.name

        data_vars[var_name] = super().to_xarray(
            encoding
        )  # use Variable to_array

        coords = self.domain.aux_coords
        if coords:
            if encoding:
                coords_names = " ".join(
                    [self.domain.coords[x].ncvar for x in coords]
                )
            else:
                coords_names = " ".join(
                    [self.domain.coords[x].name for x in coords]
                )
            data_vars[var_name].encoding["coordinates"] = coords_names

        coords = self.domain.to_xarray(encoding)

        crs_name = self.domain.grid_mapping_name
        data_vars[var_name].encoding["grid_mapping"] = crs_name

        if self.cell_methods is not None:
            data_vars[var_name].attrs["cell_methods"] = str(self.cell_methods)

        if self._ancillary is not None:
            for a in self.ancillary:
                data_vars[a] = a.to_xarray(encoding)

        # NOTE: a workaround for keeping domaintype
        # Issue: https://github.com/geokube/geokube/issues/147
        # If saved in .nc file, domain_type should be converted to str
        if (domain_type := self.domain.type) is not None and (not encoding):
            data_vars[var_name].attrs["__geo_domtype"] = domain_type

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def persist(self, path=None) -> str:
        if path is None:
            path = os.path.join(
                tempfile.gettempdir(), f"{str(uuid.uuid4())}.nc"
            )
        if os.path.isdir(path):
            path = os.path.join(path, f"{str(uuid.uuid4())}.nc")
        if not path.endswith(".nc"):
            self._LOG.warn(
                f"Provided persistance path: `{path}` has not `.nc` extension."
                " Adding automatically!"
            )
            warnings.warn(
                f"Provided persistance path: `{path}` has not `.nc` extension."
                " Adding automatically!"
            )
            path = path + ".nc"
        self.to_netcdf(path)
        return path

    def to_dict(self):
        description = None
        if self._mapping is not None and self.ncvar in self._mapping:
            var_map = self._mapping[self.ncvar]
            if "description" in var_map:
                description = var_map["description"]
            elif "name" in var_map:
                description = var_map["name"]
        if description is None:
            description = self.properties.get(
                "description", self.properties.get("long_name")
            )
        return {
            "units": str(self.units),
            "description": description,
        }

    @classmethod
    @geokube_logging
    def from_xarray(
        cls,
        ds: xr.Dataset,
        ncvar: str,
        id_pattern: Optional[str] = None,
        mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
        copy=False,
    ):
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"Expected type `xarray.Dataset` but provided `{type(ds)}`"
            )
        ds = convert_cftimes_to_numpy(ds)
        domain = Domain.from_xarray(
            ds, ncvar=ncvar, id_pattern=id_pattern, copy=copy, mapping=mapping
        )
        da = ds[ncvar].copy(copy)  # TODO: TO CHECK
        cell_methods = CellMethod.parse(da.attrs.pop("cell_methods", None))
        var = Variable.from_xarray(da, id_pattern, mapping=mapping)

        name = Variable._get_name(da, mapping=mapping, id_pattern=id_pattern)

        # We need to update `encoding` of var, as `Variable` doesn't contain `name`
        var.encoding.update(name=da.encoding.get("name", ncvar))
        # TODO ancillary variables
        field = Field(
            name=name,
            data=var.data,
            dims=var.dims,
            units=var.units,
            properties=var.properties,
            encoding=var.encoding,
            cell_methods=cell_methods,
            coords=domain,
        )
        field._id_pattern = id_pattern
        field._mapping = mapping
        field._domain._calculate_missing_lat_and_lon()
        return field

    @staticmethod
    def _update_coordinates(da: xr.DataArray, coords):
        if coords is None or len(coords) == 0:
            return
        if "coordinates" in da.attrs:
            da.attrs["coordinates"] = " ".join(
                chain([da.attrs["coordinates"]], coords)
            )
        elif "coordinates" in da.encoding:
            da.encoding["coordinates"] = " ".join(
                chain([da.encoding["coordinates"]], coords)
            )
        else:
            da.encoding["coordinates"] = " ".join(coords)
