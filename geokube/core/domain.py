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
import geokube.utils.xarray_parser as xrp
from geokube.core.axis import Axis, AxisType
from geokube.core.coord_system import (
    CoordSystem,
    CurvilinearGrid,
    RegularLatLon,
    parse_crs,
)
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.dimension import Dimension
from geokube.core.enums import LatitudeConvention, LongitudeConvention
from geokube.core.variable import Variable
from geokube.utils import util_methods
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
from geokube.utils.indexer_dict import IndexerDict
from geokube.utils.warnings import DomainWarning
from .variable import Variable

class DomainType(Enum):
    GRIDDED = "gridded"
    TIMESERIES = "timeseries"

class Domain:

    __slots__ = (
        "_coords",
        "_crs",
        "_type",
        "_axistype_to_name",
    )

    _LOG = HCubeLogger(name="Domain")

    _AUX_DIM_NAME = "_dim"

    def __init__(
        self,
        coords: Union[Mapping[Hashable, Tuple[np.ndarray, ...]], Iterable[Coordinate], Domain],
        crs: CoordSystem,
        domtype: Optional[DomainType] = None,
    ) -> None:
        if isinstance(coords, dict):
            for name, tupl in coords:
                # tupl -> (data, dims, axisType)
                self._coords[name] = Variable(data=tupl[0])
        if isinstance(coords, list):
            self._coords = {c.name: c for c in coords}
        self._crs = crs
        self._type = domtype
        self._axistype_to_name = {c.axis.atype: c.name for c in coords}

    @property
    def domtype(self):
        return self._type

    @property
    def coords(self):
        return self._coords

    @property
    def crs(self) -> CoordSystem:  # horizontal coordinate reference system
        return self._crs

    @property
    def latitude(self):
        return self[AxisType.LATITUDE]

    @property
    def longitude(self):
        return self[AxisType.LONGITUDE]

    @property
    def vertical(self):
        return self[AxisType.VERTICAL]

    @property
    def time(self):
        return self[AxisType.TIME]

    @property
    def x(self):
        return self[AxisType.X]

    @property
    def y(self):
        return self[AxisType.Y]

    @property
    def longitude_convention(self) -> LongitudeConvention:
        if AxisType.LONGITUDE in self._axistype_to_name:
            return self[AxisType.LONGITUDE].convention

    @property
    def latitude_convention(self) -> LatitudeConvention:
        if AxisType.LATITUDE in self._axistype_to_name:
            return self[AxisType.LATITUDE].convention

    @property
    def is_latitude_independent(self):
        return self[AxisType.LATITUDE].ctype is CoordinateType.INDEPENDENT

    @property
    def is_longitude_independent(self):
        return self[AxisType.LONGITUDE].ctype is CoordinateType.INDEPENDENT

    def __eq__(self, other):
        if self.crs != other.crs:
            return False
        coord_keys_eq = set(self._coords.keys()) == set(other._coords.keys())
        if not coord_keys_eq:
            return False
        for ck in self._coords.keys():
            if self._coords[ck].axis_type is AxisType.TIME:
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

    def __getitem__(self, key: Union[AxisType, str]) -> Coordinate:
        if isinstance(key, str):
            return self.coords[key]
        elif isinstance(key, AxisType):
            return self._coords[self._axistype_to_name.get(key)]         
        raise ex.HCubeTypeError(
             f"Indexing coordinates for Domain is supported only for object of types [string, AxisType]. Provided type: {type(axis)}",
             logger=self._LOG,
         )

    def __setitem__(self, key: str, value: Union[Coordinate, Variable]):
        self._coords[key] = value

    def __contains__(self, key: str):
        return (key in self._coords) or (
            AxisType.parse_type(key) in self._axistype_to_name
        )

    # def coordinate(self, axis: Union[str, Axis, AxisType]) -> Optional[Coordinate]:
    #     if isinstance(axis, str):
    #         # if we pass `latitude` and axis is called `lat`, we return the proper axis with AxisType.LATITUDE
    #         # Selection by generic should be explicit: coordinate(AxisType.GENERIC)
    #         if axis in self._coords:
    #             return self._coords[axis]
    #         if (at := AxisType.parse_type(axis)) is AxisType.GENERIC:
    #             raise ex.HCubeKeyError(f"`{axis}` not found!", logger=self._LOG)
    #         return self._coords.get(axis, self.coordinate(at))
    #     if isinstance(axis, Axis):
    #         return self._coords.get(axis.name)
    #     if isinstance(axis, AxisType):
    #         return self._coords.get(self._axistype_to_name.get(axis))
    #     raise ex.HCubeTypeError(
    #         f"Indexing coordinates for Domain is supported only for object of types [string, Axis, AxisType]. Provided type: {type(axis)}",
    #         logger=self._LOG,
    #     )

    def nbytes(self) -> int:
        return sum(coord.nbytes() for coord in self._coords)

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

        if (time_coord := self[AxisType.TIME]) is None:
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
        if coord.axis.atype == AxisType.LONGITUDE:
            if self.longitude_convention is LongitudeConvention.POSITIVE_WEST:
                range_b = (0.0, 360.0)
            else:
                range_b = (-180.0, 180.0)
        elif coord.axis.atype == AxisType.LATITUDE:
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
            dims=(
                Dimension(coord.dims[0].name, coord.axis),
                Dimension("bounds", Axis(AxisType.GENERIC, "bounds")),
            ),
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
    def from_xarray(
        cls,
        ds: xr.Dataset,
        field_name: str,
        id_pattern,
        copy: bool = False,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> "Domain":

        # cell_measures = da.encoding.get("cell_measures", da.attrs.get("cell_measures"))
        # if cell_measures:
        #     cls._LOG.info(
        #         "`cell_measure` found among encoding or attributes details. Processing..."
        #     )
        #     # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch07s02.html
        #     for (
        #         measure_type,
        #         measure_var_name,
        #     ) in util_methods.parse_cell_measures_string(cell_measures).items():
        #         coords.append(
        #             Coordinate.from_xarray_dataset(
        #                 ds=dset,
        #                 coord_name=measure_var_name,
        #                 errors=errors,
        #                 mapping=mapping,
        #             )
        #         )
        
        da = ds[field_name]
        coords = []
       
        for dim in da.dims:
            coords.append(Coordinate.from_xarray(ds, dim, id_pattern, copy, mapping))

        xr_coords = ds[field_name].attrs.pop("coordinates", ds[field_name].encoding.pop("coordinates", None))
        if xr_coords is not None:
            for coord in xr_coords.split(" "):
                coords.append(Coordinate.from_xarray(ds, coord, id_pattern, copy, mapping))       
        if "grid_mapping" in da.encoding:
            crs = parse_crs(da[da.encoding.pop("grid_mapping")])
        elif "grid_mapping" in da.attrs:
            crs = parse_crs(da[da.attrs.pop("grid_mapping")])
        else:
            crs = Domain.guess_crs(da)
        
        return Domain(coords=coords, crs=crs)

    @log_func_debug
    def to_xarray(self) -> xr.core.coordinates.DatasetCoordinates:
        grid = {}
        for coord in self._coords.values():
            grid[coord.nc_name] = coord.variable.to_xarray()
            if (bounds := coord.bounds) is not None:
                grid[bounds.nc_name] = bounds.to_xarray()
    
        if self.crs is not None:
            not_none_attrs = self.crs.as_crs_attributes()
            not_none_attrs["grid_mapping_name"] = self.crs.grid_mapping_name
            grid['crs'] = xr.DataArray(1.0, name="crs", attrs=not_none_attrs)
        return xr.Dataset(coords=grid).coords

