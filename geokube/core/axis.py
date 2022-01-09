import re
from enum import Enum
from typing import List, Optional

import cf_units as cf
import xarray as xr

import geokube.utils.exceptions as ex
from geokube.utils.hcube_logger import HCubeLogger

# Taken from https://unidata.github.io/MetPy/latest/_modules/metpy/xarray.html
coordinate_criteria_regular_expression = {
    "y": r"(y|rlat|grid_lat.*)",
    "x": r"(x|rlon|grid_lon.*)",
    "vertical": r"(soil|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric|pres|isotherm)[a-z_]*[0-9]*",
    "timedelta": r"time_delta",
    "time": r"time[0-9]*",
    "latitude": r"(x?lat[a-z0-9]*|nav_lat)",
    "longitude": r"(x?lon[a-z0-9]*|nav_lon)",
}


class AxisType(Enum):
    TIME = ("time", cf.Unit("hours since 1970-01-01", calendar="gregorian"))
    TIMEDELTA = ("timedelta", cf.Unit("hour"))
    LATITUDE = ("latitude", cf.Unit("degrees_north"))
    LONGITUDE = ("longitude", cf.Unit("degrees_east"))
    VERTICAL = ("vertical", cf.Unit("m"))
    X = ("x", cf.Unit("m"))
    Y = ("y", cf.Unit("m"))
    Z = ("z", cf.Unit("m"))
    RADIAL_AXIMUTH = ("radialAzimuth", cf.Unit("m"))
    RADIAL_ELEVATION = ("radialElevation", cf.Unit("m"))
    RADIAL_DISTANCE = ("radialDistance", cf.Unit("m"))
    GENERIC = ("generic", cf.Unit("unknown"))

    @property
    def name(self) -> str:
        return self.value[0]

    @property
    def default_units(self) -> cf.Unit:
        return self.value[1]

    @classmethod
    def get_available_names(cls) -> List[str]:
        return [a.value[0] for a in cls]

    @classmethod
    def _missing_(cls, key) -> "AxisType":
        return cls.GENERIC

    @classmethod
    def parse_type(cls, name) -> "AxisType":
        if name is None:
            return cls.generic()
        if isinstance(name, AxisType):
            return name
        try:
            return cls[name.upper() if isinstance(name, str) else name]
        except KeyError:
            for ax, regexp in coordinate_criteria_regular_expression.items():
                if re.match(regexp, name.lower(), re.IGNORECASE):
                    return cls[ax.upper()]
        return cls.generic()

    @classmethod
    def generic(cls) -> "AxisType":
        return AxisType.GENERIC


class Axis:

    _LOG = HCubeLogger(name="Axis")

    def __init__(
        self,
        atype: AxisType,
        name: Optional[str] = None,
    ):
        self._type = AxisType.parse_type(atype)
        self._name = name if name is not None else self._type.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def atype(self) -> AxisType:
        return self._type

    @property
    def default_units(self) -> cf.Unit:
        return self._type.default_units

    def __eq__(self, other):
        return (self.name == other.name) and (self._type == other._type)

    def __ne__(self, other):
        return not (self == other)

    @classmethod
    def from_xarray_dataarray(cls, da: xr.DataArray) -> "Axis":
        if not isinstance(da, xr.DataArray):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.DataArray` but provided `{type(da)}`",
                logger=cls._LOG,
            )

        _type = AxisType.parse_type(da.attrs.get("standard_name"))
        if _type is AxisType.GENERIC:
            _type = AxisType.parse_type(da.name)
        return Axis(atype=_type, name=da.name)
