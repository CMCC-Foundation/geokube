from __future__ import annotations

import re
from enum import Enum
from typing import List, Mapping, Optional

import xarray as xr

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.utils.hcube_logger import HCubeLogger

# from https://unidata.github.io/MetPy/latest/_modules/metpy/xarray.html
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
    TIME = ("time", Unit("hours since 1970-01-01", calendar="gregorian"))
    TIMEDELTA = ("timedelta", Unit("hour"))
    LATITUDE = ("latitude", Unit("degrees_north"))
    LONGITUDE = ("longitude", Unit("degrees_east"))
    VERTICAL = ("vertical", Unit("m"))
    X = ("x", Unit("m"))
    Y = ("y", Unit("m"))
    Z = ("z", Unit("m"))
    RADIAL_AXIMUTH = ("aximuth", Unit("m"))
    RADIAL_ELEVATION = ("elevation", Unit("m"))
    RADIAL_DISTANCE = ("distance", Unit("m"))
    GENERIC = ("generic", Unit("Unknown"))

    @property
    def default_unit(self) -> Unit:
        return self.value[1]

    @classmethod
    def values(cls) -> List[str]:
        return [a.value[1] for a in cls]

    @classmethod
    def parse(cls, name) -> "AxisType":
        if name is None:
            return cls.GENERIC
        if isinstance(name, AxisType):
            return name
        try:
            return cls[name.upper() if isinstance(name, str) else name]
        except KeyError:
            for ax, regexp in coordinate_criteria_regular_expression.items():
                if re.match(regexp, name.lower(), re.IGNORECASE):
                    return cls[ax.upper()]
        return cls.GENERIC

    @classmethod
    def _missing_(cls, key) -> "AxisType":
        return cls.GENERIC

class Axis:

    _LOG = HCubeLogger(name="Axis")

    def __init__(
        self,
        name: Union[str, Axis],
        axistype: Optional[Union[AxisType, str]] = None,
    ):
        if isinstance(name, Axis):
            self.copy(name)
        else:
            self._name = name
            if axistype is None:
                self._type = AxisType.parse(name)
            else:
                if isinstance(axistype, str):
                    self._type = AxisType.parse(axistype)
                else:
                    self._type = axistype

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> AxisType:
        return self._type

    @property
    def default_unit(self) -> Unit:
        return self.type.default_unit

    def __eq__(self, other):
        return (self.name == other.name) and (self.type == other.type)

    def __ne__(self, other):
        return not (self == other)

    def copy(self, other):
        self.name = other.name
        self.type = other.type