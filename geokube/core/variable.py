from html import escape
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import cf_units as cf
import dask.array as da
import numpy as np
import xarray as xr
from xarray.core.options import OPTIONS

import geokube.utils.exceptions as ex
from geokube.core.agg_mixin import AggMixin
from geokube.core.dimension import UNKNOWN_UNIT, Dimension
from geokube.utils import formatting, formatting_html, util_methods
from geokube.utils.attrs_encoding import CFAttributes, split_to_attrs_and_encoding
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger


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
        data: Union[np.ndarray, da.Array, xr.Variable],
        units: Optional[
            Union[cf.Unit, str]
        ] = None,  # if dims has one element, unit will be taken as default from dims
        dims: Optional[Union[Sequence[Dimension], Dimension]] = None,
        properties: Optional[
            Mapping[Hashable, str]
        ] = None,  # TODO: maybe as keword-arguments (**properties) rather than passing explicitly a dict?
        cf_encoding: Optional[Mapping[Hashable, str]] = None,
    ):
        if not (
            isinstance(data, np.ndarray)
            or isinstance(data, da.Array)
            or isinstance(data, xr.Variable)
        ):
            raise ex.HCubeTypeError(
                f"Expected argument of one of the following types `numpy.ndarray`, `dask.array.Array`, and `xarray.Variable` but provided {type(data)}",
                logger=self._LOG,
            )

        self._name = name
        self._dims = None
        if dims is not None:
            dims = np.array(dims, ndmin=1, dtype=Dimension)
            if len(dims) != data.ndim:
                raise ex.HCubeValueError(
                    f"Provided data have {data.ndim} dimensions but {len(dims)} Dimensions provided in `dims` argument",
                    logger=self._LOG,
                )
            self._dims = tuple(dims)

        # To avoid memory loading and copying, we set fastpath to True if xarray.Variable is passed as `data`
        self._variable = xr.Variable(
            data=data,
            dims=self.dims_names,
            fastpath=True,  # isinstance(data, xr.Variable)
        )
        if units is None and len(self._dims) == 1:
            units = self._dims[0].atype.default_units
        self._units = cf.Unit(units) if isinstance(units, str) else units
        self._properties = properties if properties else {}
        self._cf_encoding = cf_encoding if cf_encoding else {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def properties(self):
        return self._properties

    @property
    def dims(self) -> Tuple[Dimension, ...]:
        return self._dims

    @property
    def dims_names(self) -> Tuple[str, ...]:
        return tuple(x.name for x in self.dims) if self._dims is not None else ()

    @property
    def dims_axis_names(self) -> Tuple[str, ...]:
        return tuple(x.axis.name for x in self.dims) if self._dims is not None else ()

    @property
    def units(self) -> cf.Unit:
        return self._units

    @property
    def data(self):
        # Loads data!
        self._LOG.info("Loading data into memory...")
        return self._variable.data

    @property
    def values(self):
        # Loads data!
        self._LOG.info("Loading data into memory...")
        return self._variable.values

    @property
    def xr_variable(self):
        return self._variable

    def __len__(self):
        return len(self._variable)

    def __repr__(self) -> str:
        return formatting.array_repr(self.to_xarray_dataarray())

    def _repr_html_(self):
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self.to_xarray_dataarray()))}</pre>"
        return formatting_html.array_repr(self)

    def convert_units(self, unit, inplace=True):
        unit = cf.Unit(unit) if isinstance(unit, str) else unit
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

    @property
    def dtype(self):
        return self._variable.dtype

    @property
    def size(self):
        return self._variable.size

    @property
    def nbytes(self):
        return self._variable.nbytes

    @property
    def shape(self):
        return self._variable.shape

    @property
    def ndim(self):
        return self._variable.ndim

    @classmethod
    @log_func_debug
    def from_xarray_dataarray(cls, da: xr.DataArray, copy=False):
        if not isinstance(da, xr.DataArray):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.DataArray` but provided `{type(da)}`",
                logger=cls._LOG,
            )

        data = da.data.copy() if copy else da.data

        # Attributes dict should be always copied, as while poping, we remove units from original xr.DataArray
        attrs = da.attrs.copy()
        attrs.update(da.encoding)
        dims = []
        for d in da.dims:
            dims.append(Dimension.from_xarray_dataarray(da[d]))

        # For dependent coordinate like lat(rlat, rlon) -> units are in attrs not encoding
        units = Dimension._parse_units(
            da.encoding.get("units", da.attrs.get("units")),
            calendar=da.encoding.get("calendar", da.attrs.get("calendar")),
        )

        properties, cf_encoding = CFAttributes.split_attrs(attrs)
        return Variable(
            name=da.name,
            data=data,
            dims=dims,
            units=units,
            properties=properties,
            cf_encoding=cf_encoding,
        )

    @log_func_debug
    def to_tuple(self, return_variable=True):
        attrs = self.properties.copy()
        if self.units is not None and self.units != UNKNOWN_UNIT:
            if self.units.is_time_reference():
                attrs["units"] = self.units.cftime_unit
                attrs["calendar"] = self.units.calendar
            else:
                attrs["units"] = str(self.units)

        attrs.update(self._cf_encoding)
        attrs, encoding = split_to_attrs_and_encoding(attrs)
        # Sometimes (for Field) we need to return variable rather than its data in order to avoid memory issues (loading data)
        return (
            self.dims_axis_names,
            self._variable if return_variable else self._variable.data,
            attrs,
            encoding,
        )

    def to_xarray_variable(self):
        T = self.to_tuple()
        res = xr.Variable(
            data=self._variable.data, dims=self._variable.dims, fastpath=True
        )
        res.dims = T[0]
        res.attrs = T[2]
        res.encoding = T[3]
        return res

    def to_xarray_dataarray(self):
        _var = self.to_xarray_variable()
        # xarray.Variable doesn't have coordiantes, it can be accessed only by integers
        coords = {
            d: xr.Variable(data=np.arange(s), dims=(d,))
            for d, s in zip(self.dims_axis_names, self._variable.shape)
        }
        return xr.DataArray(data=_var, name=self.name, coords=coords, fastpath=True)

    def to_xarray_dataset(self):
        _da = self.to_xarray_dataarray()
        return xr.Dataset(data_vars={_da.name: _da})
