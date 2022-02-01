import functools as ft
import os
import warnings
from html import escape
from itertools import chain
from numbers import Number
from typing import Any, Callable, Hashable, List, Mapping, Optional, Tuple, Union

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cartf
import dask.array as da
import numpy as np
from geokube.core.coordinate import CoordinateType
import pyarrow as pa
import xarray as xr
import xesmf as xe
from dask import is_dask_collection
from xarray.core.options import OPTIONS

import geokube

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.core.axis import Axis
from geokube.core.cell_methods import CellMethod
from geokube.core.coord_system import CoordSystem, RegularLatLon, RotatedGeogCS
from geokube.core.axis import Axis
from geokube.core.domain import Domain
from geokube.core.coordinate import Coordinate
from geokube.core.enums import MethodType, RegridMethod
from geokube.core.variable import Variable
from geokube.utils import formatting, formatting_html, util_methods
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
from geokube.utils.indexer_dict import IndexerDict

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


class Field(Variable, DomainMixin):

    __slots__ = ("_name", "_domain", "_cell_methods", "_ancillary_data")

    _LOG = HCubeLogger(name="Field")

    def __init__(
        self,
        name: str,
        data: Union[Number, np.ndarray, da.Array, xr.Variable] = None,
        dims: Optional[Tuple[Axis]] = None,
        units: Optional[Union[Unit, str]] = None,
        properties: Optional[Mapping[Hashable, str]] = None,
        encoding: Optional[Mapping[Hashable, str]] = None,
        domain: Optional[Union[Mapping[Hashable, Any], Domain]] = None,
        cell_methods: Optional[CellMethod] = None,
        ancillary: Optional[Mapping[Hashable, Union[np.ndarray, Variable]]] = None,
    ) -> None:

        super().__init__(
            data=data, units=units, dims=dims, properties=properties, encoding=encoding
        )
        self._ancillary = None
        self._name = name
        self._domain = domain if isinstance(domain, Domain) else Domain(domain)

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
    def ancillary(self) -> Optional[Mapping[Hashable, Variable]]:
        return self._ancillary

    def __contains__(self, key):
        raise ex.HCubeNotImplementedError(
            "You cannot check inclusion for a geokube.Files", logger=Field._LOG
        )

    def __repr__(self) -> str:
        return self.to_xarray(encoding=False).__repr__()

    #        return formatting.array_repr(self.to_xarray())

    def _repr_html_(self):
        return self.to_xarray(encoding=False)._repr_html_()
        # if OPTIONS["display_style"] == "text":
        #     return f"<pre>{escape(repr(self.to_xarray()))}</pre>"
        # return formatting_html.array_repr(self)

    # geobbox and locations operates also on dependent coordinates
    # they refer only to GeoCoordinates (lat/lon)
    # TODO: Add Vertical
    @log_func_debug
    def geobbox(
        self,
        north=None,
        south=None,
        west=None,
        east=None,
        top=None,
        bottom=None,
        roll_if_needed=True,
    ):
        if (top is not None) or (bottom is not None):
            raise ex.HCubeNotImplementedError(
                "Selecting by geobbox containing vertical is currently not supported!",
                logger=self._LOG,
            )
        if not util_methods.is_atleast_one_not_none(
            north, south, west, east, top, bottom
        ):
            raise ex.HCubeKeyError(
                "At least on of the following must be defined: [north, south, west, east, top, bottom]!",
                logger=self._LOG,
            )
        return self._geobbox_idx(
            south=south,
            north=north,
            west=west,
            east=east,
            top=top,
            bottom=bottom,
            roll_if_needed=roll_if_needed,
        )

    def _geobbox_cartopy(
        self, south, north, west, east, top, bottom, roll_if_needed=True
    ):
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
            domain[Axis.LATITUDE].dims[1].axis.name: np.s_[x.min() : x.max()],
            domain[Axis.LATITUDE].dims[0].axis.name: np.s_[y.min() : y.max()],
        }
        ds = (
            self._check_and_roll_longitude(self.to_xarray(), idx)
            if roll_if_needed
            else self.to_xarray()
        )
        return Field.from_xarray_dataset(
            ds=ds.sel(indexers=idx), field_name=self.name, deep_copy=False
        )

    def _geobbox_idx(
        self, south, north, west, east, top=None, bottom=None, roll_if_needed=True
    ):
        # TODO: add vertical also
        domain = self._domain
        lat = domain.latitude.values
        lon = domain.longitude.values

        ind_lat = domain.is_latitude_independent
        ind_lon = domain.is_longitude_independent
        if ind_lat and ind_lon:
            idx = {
                domain.latitude.name: np.s_[south:north]
                if util_methods.is_nondecreasing(domain.latitude.values)
                else np.s_[north:south],
                domain.longitude.name: np.s_[west:east]
                if util_methods.is_nondecreasing(domain.longitude.values)
                else np.s_[east:west],
            }
            return self.sel(indexers=idx, roll_if_needed=roll_if_needed)

        # Specifying the mask(s) and extracting the indices that correspond to
        # the inside the bounding box.
        if np.sum(lat_mask := (lat >= float(south)) & (lat <= float(north))) == 0:
            lat_mask = (lat <= float(south)) & (lat >= float(north))
        if np.sum(lon_mask := (lon >= float(west)) & (lon <= float(east))) == 0:
            lon_mask = (lon <= float(west)) & (lon <= float(east))
        y, x = np.nonzero(lat_mask & lon_mask)

        # Spatial subseting.
        idx = {
            domain[Axis.LATITUDE].dims[1].axis.name: np.s_[x.min() : x.max() + 1],
            domain[Axis.LATITUDE].dims[0].axis.name: np.s_[y.min() : y.max() + 1],
        }
        ds = (
            self._check_and_roll_longitude(self.to_xarray(), idx)
            if roll_if_needed
            else self.to_xarray()
        )
        return Field.from_xarray(
            ds=ds.isel(indexers=idx), field_name=self.name, deep_copy=False
        )

    def locations(
        self,
        latitude,
        longitude,
        vertical: Optional[
            List[Number]
        ] = None,  # { 'latitude': [], 'longitude': [], 'vertical': []}
    ):  # points are expressed as arrays for coordinates (dep or ind) lat/lon/vertical
        # TODO: handle vertical, too
        if vertical is not None:
            raise ex.HCubeNotImplementedError(
                "Selecting by location with vertical is currently not supported!",
                logger=self._LOG,
            )
        return self._locations_idx(latitude=latitude, longitude=longitude)

    def _locations_cartopy(self, latitude, longitude):
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
            pts = domain.crs.transform_points(src_crs=plate, x=lons, y=lats)
            idx = {
                domain.x.name: xr.DataArray(data=pts[:, 0], dims="points"),
                domain.y.name: xr.DataArray(data=pts[:, 1], dims="points"),
            }

        return Field.from_xarray_dataset(
            ds=self.to_xarray().sel(indexers=idx, method="nearest"),
            field_name=self.name,
            deep_copy=False,
        )

    def _locations_idx(self, latitude, longitude):
        lats = np.array(latitude, dtype=np.float32, ndmin=1)
        lons = np.array(longitude, dtype=np.float32, ndmin=1)

        domain = self._domain
        ind_lat = domain.is_latitude_independent
        ind_lon = domain.is_longitude_independent
        if ind_lat and ind_lon:
            idx = {
                domain.latitude.name: lats.item() if len(lats) == 1 else lats,
                domain.longitude.name: lons.item() if len(lons) == 1 else lons,
            }

            return self.sel(indexers=idx, roll_if_needed=False, method="nearest")
        else:
            # Adjusting the shape of the latitude and longitude coordinates.
            # TODO: Check if these are NumPy arrays.
            # TODO: Check axes and shapes manipulation again.
            lat_coord = domain.latitude.values
            lat_dims = (np.s_[:],) + (np.newaxis,) * lat_coord.ndim
            lat_coord = lat_coord[np.newaxis, :]
            lon_coord = domain.longitude.values
            lon_dims = (np.s_[:],) + (np.newaxis,) * lon_coord.ndim
            lon_coord = lon_coord[np.newaxis, :]

            # Adjusting the shape of the latitude and longitude of the
            # locations.
            lats = lats[lat_dims]
            lons = lons[lon_dims]

            # Calculating the squares of the Euclidean distance.
            lat_diff = lat_coord - lats
            lon_diff = lon_coord - lons
            diff_sq = lat_diff * lat_diff + lon_diff * lon_diff

            # Selecting the indices that correspond to the squares of the
            # Euclidean distance.
            # TODO: Improve vectorization.
            # TODO: Consider replacing `numpy.unravel_index` with
            # `numpy.argwhere`, with the constructs like
            # `np.argwhere(diff_sq[i] == diff_sq[i].min())[0]`.
            n, *shape = diff_sq.shape
            idx = tuple(
                np.unravel_index(indices=diff_sq[i].argmin(), shape=shape)
                for i in range(n)
            )
            idx = np.array(idx, dtype=np.int_)

            # Spatial subseting.
            idx = {
                # The same order of dimensions for LATITUDE and LONGITUDE
                domain[Axis.LATITUDE]
                .dims[1]
                .axis.name: xr.DataArray(data=idx[:, 1], dims="points"),
                domain[Axis.LATITUDE]
                .dims[0]
                .axis.name: xr.DataArray(data=idx[:, 0], dims="points"),
            }

        return Field.from_xarray_dataset(
            ds=self.to_xarray().isel(indexers=idx),
            field_name=self.name,
            deep_copy=False,
        )

    # consider only independent coordinates
    # we should use metpy approach (user can also specify - units)
    @log_func_debug
    def sel(
        self,
        indexers: Mapping[Union[Axis, str], Any] = None,
        roll_if_needed: bool = True,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "Field":
        indexers = xr.core.utils.either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        indexers = indexers.copy()
        ds = self.to_xarray()

        if "time" in indexers and util_methods.is_time_combo(indexers["time"]):
            # time is always independent coordinate
            # updating with standard names of axis/axis types
            # using integer indices!!!
            ds = ds.isel(
                self.domain._process_time_combo(indexers.pop("time")), drop=drop
            )

        if roll_if_needed:
            ds = self._check_and_roll_longitude(ds, indexers)
        indexers = {self.get_netcdf_name_for_Axis(k): v for k, v in indexers.items()}
        ds = ds.sel(indexers, tolerance=tolerance, method=method, drop=drop)
        return Field.from_xarray_dataset(ds, field_name=self.name)

    @log_func_debug
    def _check_and_roll_longitude(self, ds, indexers) -> xr.Dataset:
        # `ds` here is passed as an argument to avoid one redundent to_xarray call
        if "longitude" not in indexers or not isinstance(indexers["longitude"], slice):
            return ds
        if self.domain["longitude"].ctype is not CoordinateType.INDEPENDENT:
            # TODO: implement for dependent coordinate
            raise ex.HCubeNotImplementedError(
                "Rolling longitude is currently supported only for independent coordinate!",
                logger=self._LOG,
            )
        first_el, last_el = (
            self.domain["longitude"].min(),
            self.domain["longitude"].max(),
        )

        start = indexers["longitude"].start
        stop = indexers["longitude"].stop

        sel_neg_conv = (start < 0) | (stop < 0)
        sel_pos_conv = (start > 180) | (stop > 180)

        dset_neg_conv = first_el < 0
        dset_pos_conv = first_el >= 0
        lng_name = self.domain["longitude"].name

        if dset_pos_conv and sel_neg_conv:
            # from [0,360] to [-180,180]
            roll_value = (ds[lng_name] > 180).sum().item()
            return ds.assign_coords(
                {lng_name: (((ds[lng_name] + 180) % 360) - 180)}
            ).roll(**{lng_name: roll_value}, roll_coords=True)
        if dset_neg_conv and sel_pos_conv:
            # from [-180,-180] to [0,360]
            roll_value = (ds[lng_name] < 0).sum().item()
            return ds.assign_coords({lng_name: (ds[lng_name] % 360)}).roll(
                **{lng_name: -roll_value}, roll_coords=True
            )
        return ds

    def to_regular(self):
        domain = self.domain

        # Infering latitude and longitude steps from the x and y coordinates.
        if isinstance(domain.crs, RotatedGeogCS):
            lat_step = domain.y.values.ptp() / (domain.y.values.size - 1)
            lon_step = domain.x.values.ptp() / (domain.x.values.size - 1)
        else:
            raise NotImplementedError(
                "'domain' has the coordinate reference system of the type "
                f"{type(domain.crs).__name__} that is not currently supported"
            )

        # Building regular latitude-longitude coordinates.
        south = domain.latitude.values.min()
        north = domain.latitude.values.max()
        west = domain.longitude.values.min()
        east = domain.longitude.values.max()
        lat = np.arange(south, north + lat_step / 2, lat_step)
        lon = np.arange(west, east + lon_step / 2, lon_step)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        # shape = (lat.size, lon.size)
        # lat_2d = np.broadcast_to(lat.reshape(-1, 1), shape=shape)
        # lon_2d = np.broadcast_to(lon.reshape(1, -1), shape=shape)

        # Transforming grid into regular.
        pts = domain.crs.as_cartopy_crs().transform_points(
            src_crs=ccrs.PlateCarree(), x=lon_2d, y=lat_2d
        )
        x, y = pts[:, :, 0], pts[:, :, 1]

        # Building the grid (using xarray)
        dims = [domain.latitude.nc_name, domain.longitude.nc_name]
        grid = xr.Dataset(
            data_vars={domain.x.nc_name: (dims, x), domain.y.nc_name: (dims, y)},
            coords={domain.latitude.nc_name: lat, domain.longitude.nc_name: lon},
        )

        grid[domain.latitude.nc_name].attrs = domain.latitude.variable.properties
        grid[domain.longitude.nc_name].attrs = domain.longitude.variable.properties
        grid[domain.latitude.nc_name].encoding = domain.latitude.variable.encoding
        grid[domain.longitude.nc_name].encoding = domain.longitude.variable.encoding

        # Interpolating the data.
        dset = self.to_xarray()
        dset = dset.drop(labels=[domain.latitude.nc_name, domain.longitude.nc_name])
        regrid_dset = dset.interp(
            coords={
                domain.x.nc_name: grid[domain.x.nc_name],
                domain.y.nc_name: grid[domain.y.nc_name],
            },
            method="nearest",
        )
        regrid_dset = regrid_dset.drop(labels=[domain.x.nc_name, domain.y.nc_name])
        fillValue = -9.0e-20
        regrid_dset.fillna(fillValue)
        regrid_dset[self.nc_name].encoding["_FillValue"] = fillValue
        field = Field.from_xarray(ds=regrid_dset, ncvar_name=self.nc_name, copy=False)
        field.domain._crs = RegularLatLon()

        return field

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
                "'target' must be an instance of Domain or Field", logger=self._LOG
            )

        if not isinstance(method, RegridMethod):
            method = RegridMethod[str(method).upper()]

        if reuse_weights and (weights_path is None or not os.path.exists(weights_path)):
            self._LOG.warn("`weights_path` is None or file does not exist!")
            self._LOG.info("`reuse_weights` turned off")
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
        xr_ds = self.to_xarray()
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
                logger=self._LOG,
            )
        if func is None:
            raise ex.HCubeValueError(
                f"Provided operator `{operator}` was not found! Check available operators or provide it the one yourself by pasing callable object!",
                logger=self._LOG,
            )

        # ################## Temporary solution for time bounds adjustmnent ######################

        # TODO: handle `formula_terms` bounds attribute
        # http://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries
        tb = time_axis.bounds
        ds = self.to_xarray()
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
                self._LOG.warn("Time bounds not defined for the cell methods!")
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
                self._LOG.info("`gridline` turned on")
                gridlines = True
        if gridline_labels is None:
            self._LOG.info("`gridline_labels` turned off")
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

        print(f"field {ncvar}")
        da = ds[ncvar].copy(copy)  # TODO: TO CHECK
        cell_methods = CellMethod.parse(da.attrs.pop("cell_methods", None))
        var = Variable.from_xarray(da, id_pattern, mapping=mapping)
        # We need to update `encoding` of var, as `Variable` doesn't contain `name`
        var.encoding.update(name=ncvar)

        domain = Domain.from_xarray(
            ds, ncvar=ncvar, id_pattern=id_pattern, copy=copy, mapping=mapping
        )
        name = Variable._get_name(da, mapping=mapping, id_pattern=id_pattern)
        # TODO ancillary variables
        field = Field(
            name=name,
            data=var.data,
            dims=var.dims,
            units=var.units,
            properties=var.properties,
            encoding=var.encoding,
            domain=domain,
            cell_methods=cell_methods,
        )
        print(field)
        return field
