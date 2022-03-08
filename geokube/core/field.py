from __future__ import annotations

from collections.abc import Sequence
import functools as ft
import os
import warnings
from html import escape
from itertools import chain
from numbers import Number
from typing import Any, Callable, Hashable, List, Mapping, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cartf
import dask.array as da
import numpy as np
import pyarrow as pa
import xarray as xr
import xesmf as xe
from dask import is_dask_collection
from xarray.core.options import OPTIONS

from ..utils import exceptions as ex
from ..utils import formatting, formatting_html, util_methods
from ..utils.decorators import log_func_debug
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis, AxisType
from .cell_methods import CellMethod
from .coord_system import CoordSystem, RegularLatLon, RotatedGeogCS
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
                    str, Tuple[Tuple[str, ...], Union[Number, np.ndarray, da.Array]]
                ],
            ]
        ] = None,
        crs: Optional[CoordSystem] = None,
        units: Optional[Union[Unit, str]] = None,
        properties: Optional[Mapping[Hashable, str]] = None,
        encoding: Optional[Mapping[Hashable, str]] = None,
        cell_methods: Optional[CellMethod] = None,
        ancillary: Optional[Mapping[Hashable, Union[np.ndarray, Variable]]] = None,
    ) -> None:

        super().__init__(
            data=data, units=units, dims=dims, properties=properties, encoding=encoding
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
                raise ex.HCubeTypeError(
                    f"Expected type of `ancillary` argument is dict, but the provided one if {type(ancillary)}",
                    logger=Field._LOG,
                )
            res_anc = {}
            for k, v in ancillary.items():
                if not isinstance(v, (np.ndarray, da.Array, Variable, Number)):
                    raise ex.HCubeTypeError(
                        f"Expected type of single ancillary variable is: `numpy.ndarray`, `dask.Array`, `geokube.Variable`, or `Number`, but the provided one if {type(v)}",
                        logger=Field._LOG,
                    )
                # TODO: what should be axis and dims for ancillary variables? SHould it be `Variable`?
                res_anc[k] = Variable(data=v)
            self._ancillary = res_anc

    def __str__(self) -> str:
        return f"Field {self.name}:{self.ncvar} with cell method: {self.cell_methods}"

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
    @log_func_debug
    def geobbox(
        self,
        north: Number,
        south: Number,
        west: Number,
        east: Number,
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
        HCubeKeyError
            If no bound is provided.

        """
        if not util_methods.is_atleast_one_not_none(
            north, south, west, east, top, bottom
        ):
            raise ex.HCubeKeyError(
                "At least on of the following must be defined: [north, south, west, east, top, bottom]!",
                logger=Field._LOG,
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
        lat, lon = field.latitude, field.longitude

        # Vertical
        # NOTE: In this implementation, vertical is always considered an
        # independent coordinate.
        if top is not None or bottom is not None:
            try:
                vert = field.vertical
            except ex.HCubeKeyError:
                vert = None
            # TODO: Reconsider `not vert.shape`.
            if vert is None or not vert.shape:
                raise ValueError(
                    "'top' and 'bottom' must be None because there is no "
                    "vertical coordinate or it is constant"
                )
            vert_incr = util_methods.is_nondecreasing(vert.data)
            vert_slice = np.s_[bottom:top] if vert_incr else np.s_[top:bottom]
            vert_idx = {vert.name: vert_slice}
            field = field.sel(indexers=vert_idx, roll_if_needed=True)

        if field.is_latitude_independent and field.is_longitude_independent:
            # Case of latitude and longitude being independent.
            lat_incr = util_methods.is_nondecreasing(lat.data)
            lat_slice = np.s_[south:north] if lat_incr else np.s_[north:south]
            lon_incr = util_methods.is_nondecreasing(lon.data)
            lon_slice = np.s_[west:east] if lon_incr else np.s_[east:west]
            idx = {lat.name: lat_slice, lon.name: lon_slice}
            return field.sel(indexers=idx, roll_if_needed=True)
        else:
            # Case of latitude and longitude being dependent.
            # Specifying the mask(s) and extracting the indices that correspond
            # to the inside the bounding box.
            lat_mask = (lat.data >= south) & (lat.data <= north)
            lon_mask = (lon.data >= west) & (lon.data <= east)
            # TODO: Clarify why this is required.
            if lat_mask.sum() == 0:
                lat_mask = (lat.data <= south) & (lat.data >= north)
            if lon_mask.sum() == 0:
                lon_mask = (lon.data <= float(west)) & (lon.data <= float(east))
            nonzero_idx = np.nonzero(lat_mask & lon_mask)
            idx = {
                lat.dims[i].name: np.s_[incl_idx.min() : incl_idx.max() + 1]
                for i, incl_idx in enumerate(nonzero_idx)
            }
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
        vertical: Number | Sequence[Number] | None = None
    ) -> Field:  # points are expressed as arrays for coordinates (dep or ind) lat/lon/vertical
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
                lon_2d, lat_2d = np.meshgrid(lon, lat)
                pts = self.domain.crs.as_cartopy_crs().transform_points(
                    src_crs=ccrs.PlateCarree(), x=lon_2d, y=lat_2d
                )
                x, y = pts[..., 0], pts[..., 1]
                dims = (domain.latitude.name, domain.longitude.name)
                grid = xr.Dataset(
                    data_vars={self.x.name: (dims, x), self.y.name: (dims, y)},
                    coords=domain.to_xarray(encoding=False),
                )
                dset = dset.drop(labels=(self.latitude.name, self.longitude.name))
                interp_coords = {
                    self.x.name: grid[self.x.name],
                    self.y.name: grid[self.y.name],
                }
            dset_interp = dset.interp(coords=interp_coords, method=method)
            dset_interp = dset_interp.drop(labels=[self.x.name, self.y.name])

        # dset_interp[self.name].encoding.update(dset[self.name].encoding)
        dset_interp[self.name].encoding[
            "coordinates"
        ] = f"{domain.latitude.name} {domain.longitude.name}"
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

        field.domain.type = DomainType.POINTS
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
            ds=self.to_xarray(encoding=False).sel(indexers=idx, method="nearest"),
            ncvar=self.name,
            copy=False,
            id_pattern=self._id_pattern,
            mapping=self._mapping,
        )

    def _locations_idx(self, latitude, longitude, vertical=None):
        field = self
        sel_kwa = {"roll_if_needed": False, "method": "nearest"}
        lats = np.array(latitude, dtype=np.float32).reshape(-1)
        lons = np.array(longitude, dtype=np.float32).reshape(-1)

        n = lats.size
        if lons.size != n:
            raise ValueError(
                "'latitude' and 'longitude' must have the same number of " "items"
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
            result_dset = field.to_xarray(encoding=False).isel(indexers=idx)
            result_field = Field.from_xarray(
                ds=result_dset,
                ncvar=self.name,
                copy=False,
                id_pattern=self._id_pattern,
                mapping=self._mapping,
            )
            result_field.domain.crs = RegularLatLon()

        return result_field

    # consider only independent coordinates
    # TODO: we should use metpy approach (user can also specify - units)
    @log_func_debug
    def sel(
        self,
        indexers: Mapping[Union[Axis, str], Any] = None,
        roll_if_needed: bool = True,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,  # TODO: check if should be always True or False in out case
        **indexers_kwargs: Any,
    ) -> "Field":
        indexers = xr.core.utils.either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        # TODO:
        indexers = self.domain.map_indexers(indexers)
        # TODO: indexers from this place should be always of type -> Mapping[Axis, Any]
        #   ^ it can be done in Domain
        indexers = indexers.copy()
        ds = self.to_xarray(encoding=False)

        if (
            time_ind := indexers.get(Axis("time"))
        ) is not None and util_methods.is_time_combo(time_ind):
            # time is always independent coordinate
            ds = ds.isel(self.domain._process_time_combo(time_ind), drop=drop)
            del indexers[Axis("time")]

        if roll_if_needed:
            ds = self._check_and_roll_longitude(ds, indexers)

        indexers = {self.domain[k].name: v for k, v in indexers.items()}

        # If selection by single lat/lon, coordinate is lost as it is not stored either in da.dims nor in da.attrs["coordinates"]
        # and then selecting this location from Domain fails
        ds_dims = set(ds.dims)
        ds = ds.sel(indexers, tolerance=tolerance, method=method, drop=drop)
        lost_dims = ds_dims - set(ds.dims)
        Field._update_coordinates(ds[self.name], lost_dims)
        return Field.from_xarray(
            ds, ncvar=self.name, id_pattern=self._id_pattern, mapping=self._mapping
        )

    @log_func_debug
    def _check_and_roll_longitude(self, ds, indexers) -> xr.Dataset:
        # `ds` here is passed as an argument to avoid one redundent to_xarray call
        if "longitude" not in indexers or not isinstance(indexers["longitude"], slice):
            return ds
        if self.domain[Axis("longitude")].type is not CoordinateType.INDEPENDENT:
            # TODO: implement for dependent coordinate
            raise ex.HCubeNotImplementedError(
                "Rolling longitude is currently supported only for independent coordinate!",
                logger=Field._LOG,
            )
        first_el, last_el = (
            self.domain[Axis("longitude")].min(),
            self.domain[Axis("longitude")].max(),
        )

        start = indexers[Axis("longitude")].start
        stop = indexers[Axis("longitude")].stop

        sel_neg_conv = (start < 0) | (stop < 0)
        sel_pos_conv = (start > 180) | (stop > 180)

        dset_neg_conv = first_el < 0
        dset_pos_conv = first_el >= 0
        lng_name = self.domain[Axis("longitude")].name

        if dset_pos_conv and sel_neg_conv:
            # from [0,360] to [-180,180]
            # Attributes are lost while doing `assign_coords`. They need to be reassigned (e.q. by `update`)
            roll_value = (ds[lng_name] > 180).sum().item()
            res = ds.assign_coords(
                {lng_name: (((ds[lng_name] + 180) % 360) - 180)}
            ).roll(**{lng_name: roll_value}, roll_coords=True)
            res[lng_name].attrs.update(ds[lng_name].attrs)
            # TODO: verify of there are some attrs that need to be updated (e.g. min/max value)
            return res
        if dset_neg_conv and sel_pos_conv:
            # from [-180,-180] to [0,360]
            roll_value = (ds[lng_name] < 0).sum().item()
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

    # TO CHECK
    @log_func_debug
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
        if isinstance(target, Domain):
            target_domain = target
        elif isinstance(target, Field):
            target_domain = target.domain
        else:
            raise ex.HCubeTypeError(
                "'target' must be an instance of Domain or Field", logger=Field._LOG
            )

        if not isinstance(method, RegridMethod):
            method = RegridMethod[str(method).upper()]

        if reuse_weights and (weights_path is None or not os.path.exists(weights_path)):
            Field._LOG.warn("`weights_path` is None or file does not exist!")
            Field._LOG.info("`reuse_weights` turned off")
            reuse_weights = False

        # Input domain
        lat_in = self.domain[Axis.LATITUDE]
        lon_in = self.domain[Axis.LONGITUDE]
        name_map_in = {lat_in.axis.name: "lat", lon_in.axis.name: "lon"}

        # Output domain
        lat_out = target_domain[Axis.LATITUDE]
        lon_out = target_domain[Axis.LONGITUDE]
        name_map_out = {lat_out.axis.name: "lat", lon_out.axis.name: "lon"}

        conserv_methods = {RegridMethod.CONSERVATIVE, RegridMethod.CONSERVATIVE_NORMED}
        if method in conserv_methods:
            self.domain.compute_bounds(Axis.LATITUDE)
            self.domain.compute_bounds(Axis.LONGITUDE)
            name_map_in.update(
                {lat_in.bounds.name: "lat_b", lon_in.bounds.name: "lon_b"}
            )
            target_domain.compute_bounds(lat_out.axis.name)
            target_domain.compute_bounds(lon_out.axis.name)
            name_map_out.update(
                {lat_out.bounds.name: "lat_b", lon_out.bounds.name: "lon_b"}
            )

        # Regridding
        regrid_kwa = {
            "ds_in": self.domain.to_xarray_dataset().rename(name_map_in),
            "ds_out": target_domain.to_xarray_dataset().rename(name_map_out),
            "method": method.value,
            "unmapped_to_nan": True,
            "filename": weights_path,
        }

        try:
            regridder = xe.Regridder(**regrid_kwa, reuse_weights=reuse_weights)
        except PermissionError:
            regridder = xe.Regridder(**regrid_kwa)
        xr_ds = self.to_xarray(encoding=False)
        result = regridder(xr_ds, keep_attrs=True, skipna=False)
        result = result.rename({"lat": lat_in.axis.name, "lon": lon_in.axis.name})
        result[self.variable.name].encoding = xr_ds[self.variable.name].encoding
        # After regridding those attributes are not valid!
        util_methods.clear_attributes(result, attrs="cell_measures")
        field_out = Field.from_xarray_dataset(result, field_name=self.variable.name)
        # Take `crs`` from `target_domain` as in `result` there can be still the coordinate responsible for CRS
        field_out.domain._crs = target_domain.crs
        return field_out

    # TO CHECK
    @log_func_debug
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
            Callable-object used for aggregation or string representation of a function.
            Currently supported are methods of geokube.MethodType
        frequency :  str
            Expected resampling frequency
        inplace : bool
            Indicate if operations should be done inplace or a modified copy should be returned
        Returns
        ----------
        field : Field
            The field with values after resampling procedure if inplace==True, modified copy otherwise
        Examples:
        ----------
        Resample to day frequency taking the maximum over the elements in each day:
        >>> resulting_field = field.resample(MethodType.FIRST, frequency='1D')
        Resample to 2 month frequency taking the sum (omitting NaNs) over each 2 months
        >>> resulting_field = field.resample("nansum", frequency='2M')
        """
        time_axis = self.domain[Axis.TIME]
        time = time_axis.name
        func = None
        if isinstance(operator, str):
            operator_func = MethodType(operator)
        if isinstance(operator_func, MethodType):
            func = (
                operator_func.dask_operator
                if is_dask_collection(self)
                else operator_func.numpy_operator
            )
        elif callable(operator_func):
            func = operator_func
        else:
            raise ex.HCubeTypeError(
                f"Operator can be only one of: `str`, `MethodType`, `callable`. Provided `{type(operator)}`",
                logger=Field._LOG,
            )
        if func is None:
            raise ex.HCubeValueError(
                f"Provided operator `{operator}` was not found! Check available operators or provide it the one yourself by pasing callable object!",
                logger=Field._LOG,
            )

        # ################## Temporary solution for time bounds adjustmnent ######################

        # TODO: handle `formula_terms` bounds attribute
        # http://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries
        tb = time_axis.bounds
        ds = self.to_xarray(encoding=False)
        if self.cell_methods and tb is not None:
            # `closed=right` set by default for {"M", "A", "Q", "BM", "BA", "BQ", "W"} resampling codes ("D" not included!)
            # https://github.com/pandas-dev/pandas/blob/7c48ff4409c622c582c56a5702373f726de08e96/pandas/core/resample.py#L1383
            resample_kwargs.update({"closed": "right"})
            da = ds.resample(indexer={time: frequency}, **resample_kwargs)
            new_bounds = np.empty(
                shape=(len(da.groups), 2), dtype=np.dtype("datetime64[m]")
            )
            for i, v in enumerate(da.groups.values()):
                new_bounds[i] = [tb.values[v].min(), tb.values[v].max()]
            if tb is None:
                Field._LOG.warn("Time bounds not defined for the cell methods!")
                warnings.warn("Time bounds not defined for the cell methods!")
        else:
            da = ds.resample(indexer={time: frequency}, **resample_kwargs)
            new_bounds = np.empty(
                shape=(len(da.groups), 2), dtype=np.dtype("datetime64[m]")
            )
            for i, v in enumerate(da.groups.values()):
                new_bounds[i] = [time_axis.values[v].min(), time_axis.values[v].max()]
        da = da.reduce(func=func, dim=time, keep_attrs=True)
        res = xr.Dataset(
            da,
            coords={f"{time}_bnds": ((time, "bnds"), new_bounds)},
        )
        # TODO: adjust cell_methods after resampling!

        # #########################################################################################
        return Field.from_xarray_dataset(res, field_name=self.variable.name)

    @log_func_debug
    def to_netcdf(self, path):
        self.to_xarray().to_netcdf(path=path)

    # TO CHECK
    @log_func_debug
    def plot(
        self,
        features=None,
        gridlines=None,
        gridline_labels=None,
        subplot_kwargs=None,
        projection=None,
        figsize=None,
        robust=None,
        **kwargs,
    ):
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
        crs = self._domain.crs
        dims = set()
        time = self._domain[Axis.TIME]
        if time is not None:
            dims.add(time.name)
        vert = self._domain[Axis.VERTICAL]
        if vert is not None:
            dims.add(vert.name)
        lat = self._domain[Axis.LATITUDE]
        if lat is not None:
            dims.add(lat.name)
            kwargs.setdefault("y", lat.name)
        lon = self._domain[Axis.LONGITUDE]
        if lon is not None:
            dims.add(lon.name)
            kwargs.setdefault("x", lon.name)
        n_dims = len(dims)
        transform = crs.as_cartopy_projection() if crs is not None else None
        plate = ccrs.PlateCarree

        if n_dims in {3, 4}:
            # n_cols = None
            if time is not None and time.name in dims and time.values.size > 1:
                kwargs.setdefault("col", time.name)
                # if time.size == 4:
                #     n_cols = 2
                # if time.size >= 5:
                #     n_cols = 3
            if vert is not None and vert.name in dims and vert.values.size > 1:
                kwargs.setdefault("row", vert.name)
            # elif all(('col' in kwargs, n_cols, not has_cartopy_items)):
            #     kwargs.setdefault('col_wrap', n_cols)

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
        darr = self.to_xarray()
        if isinstance(darr, xr.Dataset):
            darr = darr[self.name]
        plot = darr.plot(**kwargs)

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
                coords = darr.coords

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

        return plot

    @log_func_debug
    def to_xarray(self, encoding=True) -> xr.Dataset:
        data_vars = {}
        var_name = self.ncvar if encoding else self.name

        data_vars[var_name] = super().to_xarray(encoding)  # use Variable to_array

        coords = self.domain.aux_coords

        if coords:
            if encoding:
                coords_names = " ".join([self.domain.coords[x].ncvar for x in coords])
            else:
                coords_names = " ".join([self.domain.coords[x].name for x in coords])
            data_vars[var_name].encoding["coordinates"] = coords_names

        coords = self.domain.to_xarray(encoding)

        data_vars[var_name].encoding["grid_mapping"] = "crs"

        if self.cell_methods is not None:
            data_vars[var_name].attrs["cell_methods"] = str(self.cell_methods)

        if self._ancillary is not None:
            for a in self.ancillary:
                data_vars[a] = a.to_xarray(encoding)

        return xr.Dataset(data_vars=data_vars, coords=coords)

    @classmethod
    @log_func_debug
    def from_xarray(
        cls,
        ds: xr.Dataset,
        ncvar: str,
        id_pattern: Optional[str] = None,
        mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
        copy=False,
    ):
        if not isinstance(ds, xr.Dataset):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.Dataset` but provided `{type(ds)}`",
                logger=cls._LOG,
            )

        da = ds[ncvar].copy(copy)  # TODO: TO CHECK
        cell_methods = CellMethod.parse(da.attrs.pop("cell_methods", None))
        var = Variable.from_xarray(da, id_pattern, mapping=mapping)
        # We need to update `encoding` of var, as `Variable` doesn't contain `name`

        domain = Domain.from_xarray(
            ds, ncvar=ncvar, id_pattern=id_pattern, copy=copy, mapping=mapping
        )
        name = Variable._get_name(da, mapping=mapping, id_pattern=id_pattern)

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
        return field

    @staticmethod
    def _update_coordinates(da: xr.DataArray, coords):
        if coords is None or len(coords) == 0:
            return
        if "coordinates" in da.attrs:
            da.attrs["coordinates"] = " ".join(chain([da.attrs["coordinates"]], coords))
        elif "coordinates" in da.encoding:
            da.encoding["coordinates"] = " ".join(
                chain([da.encoding["coordinates"]], coords)
            )
        else:
            da.encoding["coordinates"] = " ".join(coords)
