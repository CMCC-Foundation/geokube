from collections.abc import Sequence, Mapping
import os
from types import ModuleType

from typing import Any, Literal, Union


# FIXME: There is an error related to `DriverEntrypoint`.


# pylint: disable=unused-import, no-name-in-module
from . import argo, cfnetcdf, sentinel2  # noqa
# from .common import DriverEntrypoint
from geokube.core.collection import Collection
from geokube.core.field import Field
from geokube.core.cube import Cube


T_Driver = Union[
    Literal["cfnetcdf", "sentinel2", "argo"],
    # type[DriverEntrypoint],
    str,  # no nice typing support for custom backends
    None,
]

# DRIVERS = {
#     "cfnetcdf": drivers.CFNetCDF.open,
#     "nemo": drivers.Nemo.open,
#     "wrf": drivers.Wrf.open,
#     "sentinel2": drivers.Sentinel2.open,
#     "argo": drivers.Argo.open
# }


def open_fields(
    filename_or_obj: str | os.PathLike[Any],
    ncvars: str | None = None,
    driver: T_Driver = None,
    driver_kwargs: dict[str, Any] | None = None,
    xarray_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> Field | Mapping[str, Field]:
    pass


def open_cubes(
    filename_or_obj: str | os.PathLike[Any], 
    ncvars: list[str] | None = None,
    driver: T_Driver = None,
    driver_kwargs: dict[str, Any] | None = None,
    xarray_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> Cube | Sequence[Cube]:
    pass


def open_collection(
    pattern_or_obj: str,
    *,
    driver: T_Driver = None,
    driver_kwargs: dict[str, Any] | None = None,
    xarray_kwargs: dict[str, Any] | None = None,
    **kwargs,        
) -> Collection:
    if callable(driver):
        driver_call = driver
    else:
        match driver:
            case ModuleType():
                driver_module = driver
            case str():
                driver_module = globals()[driver]
            case _:
                raise TypeError(f"'driver' type {type(driver)} not supported")
        driver_call = getattr(driver_module, 'open')

    kwa = driver_kwargs or {}
    result = driver_call(pattern_or_obj, **kwa, xarray_kwargs=xarray_kwargs)
    match result:
        case Field():
            return Collection(data=[Cube(fields=[result])])
        case Cube():
            return Collection(data=[result])
        case Collection():
            return result
        case _:
            raise TypeError("'result' must be a field, cube, or collection")


# coll = geokube.open_collection(
#     '/path/to/files', 
#     driver='wrf', 
#     driver_kwargs={ 'projection': 'lmabert'},
#     xarray_kwargs={ 'engine': 'zarr'}
# )
