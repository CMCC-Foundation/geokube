from __future__ import annotations

import re
from enum import Enum
from typing import Any, Hashable, List, Mapping, Optional, Union
from geokube.utils.type_utils import OptStrMapType

import xarray as xr

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.utils.hcube_logger import HCubeLogger
from geokube.core.cfobject_mixin import CFObjectMixin

AxisStrType = Union["Axis", str]

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

    @property
    def axis_type_name(self) -> str:
        return self.value[0]

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


class Axis(CFObjectMixin):

    __slots__ = ("_name", "_type", "_encoding", "_is_dim")

    _LOG = HCubeLogger(name="Axis")

    def __init__(
        self,
        name: Union[str, Axis],
        axistype: Optional[Union[AxisType, str]] = None,
        encoding: OptStrMapType = None,
        is_dim: Optional[bool] = False,
    ):
        if isinstance(name, Axis):
            super().apply_from_other(name)
        else:
            self._is_dim = is_dim
            self._name = name
            self._encoding = encoding
            if axistype is None:
                self._type = AxisType.parse(name)
            else:
                if isinstance(axistype, str):
                    self._type = AxisType.parse(axistype)
                elif isinstance(axistype, AxisType):
                    self._type = axistype
                else:
                    raise ex.HCubeTypeError(
                        f"Expected argument is one of the following types `str`, `geokube.AxisType`, but provided {type(axistype)}",
                        logger=Axis._LOG,
                    )

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> AxisType:
        return self._type

    @property
    def default_unit(self) -> Unit:
        return self._type.default_unit

    @property
    def ncvar(self):
        return self._encoding.get("name", self.name) if self._encoding else self.name

    @property
    def encoding(self):
        return self._encoding

    @property
    def is_dim(self):
        return self._is_dimension

    def __eq__(self, other):
        return (self.name == other.name) and (self.type == other.type)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self) -> str:
        return f"<Axis(name={self.name}, type:{self.type}, encoding={self._encoding}>"

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"
