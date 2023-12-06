import os

from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    Mapping,
    Literal,
    Union,
)

from geokube.core.field import Field
from geokube.core.cube import Cube
from geokube.core.collection import Collection

if TYPE_CHECKING:

    from geokube.drivers.common import DriverEntrypoint
    # from geokube.core.types import (
    # )

    T_Driver = Union[
        Literal["cfnetcf", "nemo", "wrf", "sentinel2", "argo"],
        type[DriverEntrypoint],
        str,  # no nice typing support for custom backends
        None,
    ]

DRIVERS = {
    "cfnetcdf": drivers.CFNetCDF.open,
    "nemo": drivers.Nemo.open,
    "wrf": drivers.Wrf.open,
    "sentinel2": drivers.Sentinel2.open,
    "argo": drivers.Argo.open
}

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
    pass


coll = geokube.open_collection(
    '/path/to/files', 
    driver='wrf', 
    driver_kwargs={ 'projection': 'lmabert'},
    xarray_kwargs={ 'engine': 'zarr'}
)