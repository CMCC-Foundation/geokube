from datetime import datetime
import re
from collections.abc import Iterable
from itertools import product
from numbers import Number
from string import Formatter, Template
from typing import Any, Hashable, Mapping, Union

import numpy as np
import xarray as xr
from dask import is_dask_collection

from ..core.axis import Axis


def convert_cftimes_to_numpy(obj):
    for key in obj.coords:
        if obj[key].dtype == np.dtype("O"):
            try:
                obj[key] = obj.indexes[key].to_datetimeindex()
            except AttributeError:
                pass
            except KeyError:
                # NOTE: we skipp non-indexable cooridnates, like time_bnds
                pass            
    return obj


def are_dims_compliant(provided_shape, expected_shape):
    return provided_shape == expected_shape


def trim_key(mapping: dict, exclude: list):
    return {
        k: v for k, v in mapping.items() if k not in np.array(exclude, ndmin=1)
    }


def cast_attr_to_valid(
    attrs: Mapping[Hashable, Any]
) -> Mapping[Hashable, Union[str, Number, np.ndarray, list, tuple]]:
    return {
        k: str(v) if isinstance(v, np.dtype) else v for k, v in attrs.items()
    }


def clear_attributes(d: Union[xr.Dataset, xr.DataArray], attrs):
    attrs = np.array(attrs, ndmin=1)
    if isinstance(d, xr.DataArray):
        for a in attrs:
            d.attrs.pop(a, None)
    elif isinstance(d, xr.Dataset):
        for dv in d.data_vars:
            for a in attrs:
                d[dv].attrs.pop(a, None)


def get_indexer_list_product(indexers):
    dim_names = list(indexers.keys())
    cart_prod = product(*[np.array(v, ndmin=1) for v in indexers.values()])
    return [{d: c for d, c in zip(dim_names, c)} for c in cart_prod]


def get_not_nones(*args) -> list:
    return list(filter(lambda x: x is not None, args))


def is_atleast_one_not_none(*args, **kwargs):
    for v in args:
        if v is not None:
            return True
    for v in kwargs.values():
        if v is not None:
            return True
    return False


def assert_exactly_one_arg(**kwargs):
    provided = False
    for v in kwargs.values():
        if provided and v is not None:
            raise ValueError(
                "Those arguments cannot be passed at ones:"
                f" {list(kwargs.keys())}"
            )
        if not provided and v is not None:
            provided = True
    if not provided:
        raise ValueError(
            f"Exactly one argument must be provided: {list(kwargs.keys())}"
        )


def _agnostic_operator(method, value):
    if is_dask_collection(value):
        return method(value.compute())
    return method(value)


def agnostic_max(value):
    return _agnostic_operator(np.max, value)


def agnostic_min(value):
    return _agnostic_operator(np.min, value)


def agnostic_diff(value):
    return _agnostic_operator(np.diff, value)


def agnostic_unique(value):
    return _agnostic_operator(np.unique, value)


def is_time_combo_key(indexer: dict):
    return (
        isinstance(indexer, dict)
        and (indexer is not None)
        and (
            ("hour" in indexer.keys())
            or ("day" in indexer.keys())
            or ("month" in indexer.keys())
            or ("year" in indexer.keys())
        )
    )


def make_compatible_data(data):
    if isinstance(data, np.ndarray):
        # Necessery as np.ndarray is sometimes treated as memview class, which cannot be processed by xarray!
        return np.asarray(data)
    return data


def slice_common_part(s1, s2):
    start = np.max([np.min([s1.start, s1.stop]), np.min([s2.start, s2.stop])])
    stop = np.min([np.max([s1.start, s1.stop]), np.max([s2.start, s2.stop])])
    return slice(start, stop)


def list_to_slice_or_array(val_list):
    if isinstance(val_list, slice):
        return val_list
    elif isinstance(val_list, set):
        return list(val_list)
    elif isinstance(val_list, Iterable):
        if len(val_list) == 2:
            return val_list
        arr = np.array(list(val_list))
        if len(np.unique(df := np.diff(arr))) == 1:
            return slice(arr[0], arr[-1] + df[0], df[0])
        return arr
    return val_list


def parse_cell_measures_string(cell_measures_str) -> dict:
    return {
        vi.strip(): vi.strip()
        for v in re.findall(r"[a-z]*[:].?[a-z]*", cell_measures_str)
        for vi in v.split(":")
    }


def _is_time_part_name(name: str) -> bool:
    return name == "year" or name == "month" or name == "day" or name == "hour"


def find_slice_dict_and_change_to_slice(dict_val: Mapping[str, Any]):
    if not isinstance(dict_val, dict):
        return dict_val
    res = {}
    for k, v in dict_val.items():
        if isinstance(v, Iterable) and (("start" in v) or ("stop" in v)):
            res[k] = slice(v["start"], v["stop"], v.get("step"))
        elif isinstance(v, dict):
            res[k] = find_slice_dict_and_change_to_slice(v)
        else:
            res[k] = v

    return res


def is_time_combo(vals: dict) -> bool:
    return isinstance(vals, dict) and (
        "year" in vals or "month" in vals or "day" in vals or "hour" in vals
    )


def is_nondecreasing(values):
    return np.all(np.diff(values) >= 0)


def get_indexer_type(col):
    is_slice = False
    is_number = False
    for item in col:
        if isinstance(item, slice):
            is_slice = True
            if is_number:
                return "combo"

        if isinstance(item, Number):
            is_number = True
            if is_slice:
                return "combo"

    return "slice" if is_slice else "numeric"


def geokube_properties_to_xarray_attrs(properties, cell_methods=None):
    attrs = properties.copy()
    if "cf_name" in attrs:
        attrs["standard_name"] = attrs.pop("cf_name")
    if "description" in attrs:
        attrs["long_name"] = attrs.pop("description")
    if cell_methods:
        attrs["cell_methods"] = cell_methods.encode_for_netcdf()
    return attrs


def is_between(data, lower_bound=None, upper_bound=None):
    if lower_bound is not None:
        if upper_bound is not None:
            return (data >= lower_bound) & (data <= upper_bound)
        else:
            return data >= lower_bound
    else:
        if upper_bound is not None:
            return data <= upper_bound
        else:
            return np.full_like(data, fill_value=True, dtype=np.bool_)
