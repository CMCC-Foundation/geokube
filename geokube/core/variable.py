from __future__ import annotations

from html import escape
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
from dask.array.core import _elemwise_handle_where
import numpy as np
import xarray as xr
from xarray.core.options import OPTIONS

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.core.dimension import Dimension
from geokube.utils import formatting, formatting_html, util_methods
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
import geokube.utils.xarray_parser as xrp

class Variable(xr.Variable):

    __slots__ = (
        "__dims__",
        "_units",
    )

    _LOG = HCubeLogger(name="Variable")

    def _as_dimension_tuple(self, dims):
        if isinstance(dims, str):
            _dims = Dimension(str)
        elif isinstance(dims, tuple) or isinstance(dims, list):
            _dims = []
            for d in dims:
                _dims.append(Dimension(d))
                # if isinstance(d, str):
                #     _dims.append(Dimension(d))
                # el:
                #     _dims.append(d)
        else:
            raise ex.HCubeValueError(
                    f"not valid type for dimensions provided in `dims` argument",
                    logger=self._LOG,
            )

        return tuple(_dims)

    def __init__(
        self,
        data: Union[np.ndarray, da.Array],
        dims: Optional[Union[Sequence[Union[Dimension, str]], Union[Dimension,str]]] = None,
        units: Optional[Union[Unit, str]] = None,
        properties: Optional[ Mapping[Hashable, str]] = None, 
        encoding: Optional[Mapping[Hashable, str]] = None,
    ):
        if not (
            isinstance(data, np.ndarray)
            or isinstance(data, da.Array)
            or isinstance(data, Variable)
        ):
            raise ex.HCubeTypeError(
                f"Expected argument of one of the following types `numpy.ndarray`, `dask.array.Array`, but provided {type(data)}",
                logger=self._LOG,
            )

        if isinstance(data, Variable):
            self.copy(data)
        else:
            self.__dims__ = None
            if dims is not None:
                dims = self._as_dimension_tuple(dims)
                dims = np.array(dims, ndmin=1, dtype=Dimension)
                if len(dims) != data.ndim:
                     raise ex.HCubeValueError(
                         f"Provided data have {data.ndim} dimensions but {len(dims)} Dimensions provided in `dims` argument",
                         logger=self._LOG,
                     )
            
                self.__dims__ = dims
            super().__init__(data=data, dims=self.dim_names, attrs=properties, encoding=encoding, fastpath=True)      
            self._units = Unit(units) if (isinstance(units, str) or units is None) else units

    def dims(self) -> Tuple[Dimension, ...]:
        return self.__dims__

    @property
    def dim_names(self):
        return tuple([d.name for d in self.__dims__])

    @property
    def dim_ncvars(self):
        return tuple([d.ncvar for d in self.__dims__])
        
    @property
    def dim_axis(self):
        return tuple([d.axis for d in self.__dims__])
        
    @property
    def properties(self):
        return self.attrs

    @property
    def units(self) -> Unit:
        return self._units

    def copy(self, other, deep_copy=False):
        self._dims = other.dims
        self._units = other.units       
        super().__init__(data=other.data, dims=other.dim_names, attrs=other.properties, encoding=other.encoding)

    def __repr__(self) -> str:
        return self.to_xarray(encoding=False).__repr__()

    def _repr_html_(self):
        return self.to_xarray(encoding=False)._repr_html_()

    def convert_units(self, unit, inplace=True):
        unit = Unit(unit) if isinstance(unit, str) else unit
        if not isinstance(self.data, np.ndarray):
            self._LOG.warn(
                "Converting units is supported only for np.ndarray inner data type. Data will be loaded into the memory!"
            )
            self.data = np.array(
                self.data
            )  # TODO: inplace for cf.Unit doesn't work!
        res = self.units.convert(self.data, unit, inplace)
        if not inplace:
            return Variable(
                data=res,
                dims=self.dims,
                units=unit,
                properties=self.properties,
                encoding=self.encoding,
            )
        self.data = res
        self.units = unit

    @classmethod
    @log_func_debug
    def _get_name(cls, da, mapping, id_pattern):
        name = da.attrs.get("standard_name", da.name)
        if mapping is not None and da.name in mapping:
            name = mapping[da.name]['name']
        elif id_pattern is not None:
            name = xrp.form_id(id_pattern, da.attrs)                
        return name
    
    @classmethod
    @log_func_debug
    def from_xarray(
        cls,
        da: xr.DataArray,
        id_pattern,
        copy=False,
        mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    ):
        if not isinstance(da, xr.DataArray):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.DataArray` but provided `{type(da)}`",
                logger=cls._LOG,
            )
 
        data = da.data.copy() if copy else da.data
        dims = []
        for d in da.dims:
            if d in da.coords:
                d_name = Variable._get_name(da[d], mapping, id_pattern)
                d_axis = da[d].attrs.get("axis", None)
                dims.append(Dimension(name=d_name, axistype=d_axis, encoding={'name': d}))   
            else:
                dims.append(Dimension(name=d))

        dims = tuple(dims)
        attrs = da.attrs.copy()
        encoding = da.encoding.copy()

        units = Unit(
            encoding.pop("units", attrs.pop("units", None)),
            calendar=encoding.pop("calendar", attrs.pop("calendar", None)),
        )
                
        return Variable(
            data=data,
            dims=dims,
            units=units,
            properties=attrs,
            encoding=encoding,
        )

    @log_func_debug
    def to_xarray(self, encoding=True):
        nc_attrs = self.properties
        nc_encoding = self.encoding
        if encoding:
            dims = self.dim_ncvars
            if self.units is not None and not self.units.is_unknown:
                 if self.units.is_time_reference():
                     nc_encoding["units"] = self.units.cftime_unit
                     nc_encoding["calendar"] = self.units.calendar
                 else:
                     nc_attrs["units"] = str(self.units)
        else:
            dims = self.dim_names

        return xr.Variable(data=self._data, dims = dims, attrs=nc_attrs, encoding=nc_encoding)