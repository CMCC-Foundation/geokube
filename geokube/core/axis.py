from __future__ import annotations

import re
from enum import Enum
from typing import Any, Hashable, List, Mapping, Optional, Union

import xarray as xr

from .unit import Unit
from ..utils.hcube_logger import HCubeLogger

# from https://unidata.github.io/MetPy/latest/_modules/metpy/xarray.html
coordinate_criteria_regular_expression = {
    "y": r"(y|rlat|grid_lat.*|j)",
    "x": r"(x|rlon|grid_lon.*|i)",
    "vertical": r"(soil|lv_|bottom_top|sigma|h(ei)?ght|altitude|dept(h)?|isobaric|pres|isotherm|model_level_number)[a-z_]*[0-9]*",
    "timedelta": r"time_delta",
    "time": r"(time[0-9]*|T)",
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
        if isinstance(name, Axis):
            return name.type
        try:
            res = cls[name.upper() if isinstance(name, str) else name]
            if res is AxisType.Z:
                return AxisType.VERTICAL
            return res
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
        encoding: Optional[Mapping[Hashable, str]] = None,
        is_dim: Optional[bool] = False,
    ):
        if isinstance(name, Axis):
            self._name = name._name
            self._type = name._type
            self._encoding = name._encoding
            self._is_dim = name._is_dim
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
                    raise TypeError(
                        "Expected argument is one of the following types"
                        " `str`, `geokube.AxisType`, but provided"
                        f" {type(axistype)}"
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
        return (
            self._encoding.get("name", self.name)
            if self._encoding
            else self.name
        )

    @property
    def encoding(self):
        return self._encoding

    @property
    def is_dim(self):
        return self._is_dim

    def __hash__(self):
        enc_keys = (
            tuple(self._encoding.keys())
            if self._encoding is not None
            else tuple()
        )
        return hash((self._name, self._type, enc_keys, self._is_dim))

    def __eq__(self, other):
        if issubclass(type(other), Axis):
            return (
                (self.name == other.name)
                and (self.type == other.type)
                and (self._is_dim == other.is_dim)
                and (self._encoding == other._encoding)
            )
        return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self) -> str:
        return (
            f"<Axis(name={self.name}, type:{self.type},"
            f" encoding={self._encoding}>"
        )

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"

    @classmethod
    def get_name_for_object(cls, obj: Union[str, Axis, AxisType]) -> str:
        if isinstance(obj, Axis):
            return obj.name
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, AxisType):
            return obj.axis_type_name
        else:
            raise TypeError(
                "`dims` can be a tuple or a list of [geokube.Axis,"
                f" geokube.AxisType, str], but provided type is `{type(obj)}`"
            )
