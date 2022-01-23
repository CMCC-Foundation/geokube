import logging
import re
from itertools import chain
from string import Formatter, Template
from typing import Iterable, Mapping, Optional, Tuple, Union

import intake.source.utils
import numpy as np
import xarray as xr

import geokube.core.coord_system as crs

BOUNDS_PATTERN = re.compile(r".*(bnds|bounds).*$", re.IGNORECASE)

logger = logging.getLogger(__name__)


def get_bounds_names(names: Iterable):
    return [n for n in names if BOUNDS_PATTERN.search(n)]


def get_crs(
    coords: Mapping[str, xr.DataArray], default: crs.CoordSystem = crs.RegularLatLon()
) -> Tuple[str, crs.CoordSystem]:
    for k, v in coords.items():
        grid_name = v.attrs.get("grid_mapping_name")
        if not grid_name:
            continue
        proj_attrs = v.attrs.copy()
        del proj_attrs["grid_mapping_name"]
        if coord := crs.get_coord_system(grid_name, **proj_attrs):
            return (k, coord)
    return (default.grid_mapping_name, default)


def is_bounds(da: xr.DataArray):
    return BOUNDS_PATTERN.search(da.name) is None


def get_coord_name_for_bound_name_by_dimensions(
    da: Union[xr.DataArray, xr.Dataset], name: str
):
    # Last dimension of bounds is bnds or axis - [0,1]
    dims = np.array(da[name].dims[:-1], ndmin=1, dtype=str)
    return {d: name for d in dims}


def get_coord_name_for_bound_name(name: str):
    bn = BOUNDS_PATTERN.search(name)
    if bn is not None:
        return intake.source.utils.reverse_format("{coord}_{}", bn.group(0))["coord"]


def get_coords_by_bounds(bounds_names: list):
    return [
        intake.source.utils.reverse_format("{coord}_{}", _)["coord"]
        for _ in bounds_names
    ]


def get_global_ancillary_variables(data: xr.Dataset):
    return list(
        set(filter(lambda c: len(data[c].dims) == 0, list(data.data_vars.keys())))
    )


def get_ancillary_variables(data: xr.DataArray):
    ancill_vars = data.attrs.get("ancillary_variables")
    if ancill_vars:
        return ancill_vars.split(" ")


def get_coords_by_bounds(bounds_names: list):
    return [
        intake.source.utils.reverse_format("{coord}_{}", _)["coord"]
        for _ in bounds_names
    ]


def get_auxiliary_coords(dataset: xr.Dataset):
    # Non-auxillary coordinates depend on their owns: lat(lat), lon(lon)
    # Auxilary depends on dimensions lat(x,y), lon(x,y), lat(rlat, rlon)
    # If 1st dimension of coordinate has the same name as a coordinate itself
    # then it is an oridnary coordinate-dimension. Otherwise, it is an auxillary coordinate
    return set([k for k, d in dataset.coords.items() if d.dims and d.dims[0] != k])


def form_id(id_pattern, attrs):
    fmt = Formatter()
    _, field_names, _, _ = zip(*fmt.parse(id_pattern))
    field_names = [_ for _ in field_names if _]
    # Replace intake-like placeholder to string.Template-like ones
    for k in field_names:
        if k not in attrs:
            raise KeyError(
                f"Requested id component - `{k}` is not present in attributes!"
            )
        id_pattern = id_pattern.replace(
            f"{{{k}}}", f"${{{k}}}"
        )  # "{some_field}" -> "${some_field}"
    template = Template(id_pattern)
    return template.substitute(**{k: attrs[k] for k in field_names})
