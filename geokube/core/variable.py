from html import escape
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from xarray.core.options import OPTIONS

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.core.agg_mixin import AggMixin
from geokube.core.dimension import Dimension
from geokube.utils import formatting, formatting_html, util_methods
from geokube.utils.attrs_encoding import CFAttributes, split_to_xr_attrs_and_encoding
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
import geokube.utils.xarray_parser as xrp

class Variable(AggMixin):

    __slots__ = (
        "_name",
        "_dims",
        "_variable",
        "_units",
        "_properties",
        "_cf_encoding",
    )

    _LOG = HCubeLogger(name="Variable")

    def __init__(
        self,
        name: str,
        data: Union[np.ndarray, da.Array],
        units: Optional[ Union[Unit, str]] = None,
        dims: Optional[Union[Sequence[str], str]] = None,
        properties: Optional[ Mapping[Hashable, str]] = None, 
        encoding: Optional[Mapping[Hashable, str]] = None,
    ):
        if not (
            isinstance(data, np.ndarray)
            or isinstance(data, da.Array)
        ):
            raise ex.HCubeTypeError(
                f"Expected argument of one of the following types `numpy.ndarray`, `dask.array.Array`, but provided {type(data)}",
                logger=self._LOG,
            )

        self._name = name
        self._data = data
        self._dims = None
        if dims is not None:
            dims = np.array(dims, ndmin=1, dtype=str)
            if len(dims) != data.ndim:
                raise ex.HCubeValueError(
                    f"Provided data have {data.ndim} dimensions but {len(dims)} Dimensions provided in `dims` argument",
                    logger=self._LOG,
                )
            self._dims = tuple(dims)
        self._encoding = encoding if encoding else {}
        self._units = Unit(units) if (isinstance(units, str) or None) else units
        self._properties = properties if properties else {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def nc_name(self) -> str:
        return self._encoding.get('name', self.name)

    @property
    def nc_dims(self) -> str:
        return self._encoding.get('dims', self.dims)

    @property
    def properties(self):
        return self._properties

    @property
    def encoding(self):
        return self._encoding

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._dims

    @property
    def units(self) -> Unit:
        return self._units

    @property
    def data(self):
        return self._data.data

    @property
    def values(self):
        return self._data.values

    def __len__(self):
        return len(self._variable)

    def __repr__(self) -> str:
        return formatting.array_repr(self.to_xarray_dataarray())

    def _repr_html_(self):
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self.to_xarray_dataarray()))}</pre>"
        return formatting_html.array_repr(self)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self):
        return self._data.size

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def convert_units(self, unit, inplace=True):
        unit = Unit(unit) if isinstance(unit, str) else unit
        if not isinstance(self.data, np.ndarray):
            self._LOG.warn(
                "Converting units is supported only for np.ndarray inner data type. Data will be loaded into the memory!"
            )
            self._variable.data = np.array(
                self._variable.data
            )  # TODO: inplace for cf.Unit doesn't work!
        res = self.units.convert(self.data, unit, inplace)
        if not inplace:
            return Variable(
                name=self._name,
                data=res,
                dims=self.dims,
                units=unit,
                properties=self._properties,
                cf_encoding=self._cf_encoding,
            )
        self._variable.data = res
        self._units = unit

    @staticmethod
    def _get_var_name(
        da: xr.DataArray,
        field_id: Optional[str] = None,
        mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    ):
        name = da.attrs.get("standard_name", da.name) # by default standard name
        props = {}
        # Mapping has higher priority than field_id
        if mapping is not None and da.name in mapping:
            name = mapping[da.name]["name"]
            props = util_methods.trim_key(mapping[da.name], exclude="api")
        elif field_id is not None:
            name = xrp.form_field_id(field_id, da.attrs)
        return name, props

    @classmethod
    @log_func_debug
    def _get_var_name( cls, da, mapping, id_pattern):
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
        name = Variable._get_var_name(da, mapping, id_pattern)
        print(f"from xarray name {name}")
        data = da.data.copy() if copy else da.data
        dims = []
        for d in da.dims:
            dims.append(Variable._get_var_name(da[d], mapping, id_pattern))

        attrs = da.attrs.copy()
        encoding = da.encoding.copy()

        units = Dimension._parse_units(
            encoding.pop("units", attrs.pop("units", None)),
            calendar=encoding.pop("calendar", attrs.pop("calendar", None)),
        )
        
        encoding['name'] = da.name
        encoding['dims'] = da.dims
        
        # pop coordinates, grid_mapping_name, cell_measures
        attrs.pop("coordinates", None)
        attrs.pop("grid_mapping_name", None)
        attrs.pop("grid_mapping", None)
        attrs.pop("cell_measures", None)        
        attrs.pop("cell_methods", None)        
        encoding.pop("coordinates", None)
        encoding.pop("grid_mapping_name", None)
        encoding.pop("grid_mapping", None)
        encoding.pop("cell_measures", None)        
        encoding.pop("cell_methods", None)        
        
        return Variable(
            name=name,
            data=data,
            dims=dims,
            units=units,
            properties=attrs,
            encoding=encoding,
        )

    @log_func_debug
    def to_xarray(self):
        xr_attrs = self.properties
        # if self.units is not None and not self.units.is_unknown:
        #     if self.units.is_time_reference():
        #         xr_attrs["units"] = self.units.cftime_unit
        #         xr_attrs["calendar"] = self.units.calendar
        #     else:
        #         xr_attrs["units"] = str(self.units)
        xr_encoding = self.encoding
        return xr.Variable(data=self._data, dims = self.nc_dims, attrs=xr_attrs, encoding=xr_encoding)