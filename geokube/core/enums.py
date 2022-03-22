import re
from contextlib import suppress
from enum import Enum, unique
from typing import Optional, Union

import dask.array as da
import numpy as np


class MethodType(Enum):
    # From https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ape.html
    POINT = ("point", [None])
    MAX = ("max", [da.nanmax, np.nanmax])
    MIN = ("min", [da.nanmin, np.nanmin])
    MEAN = ("mean", [da.nanmean, np.nanmean])
    SUM = ("sum", [da.nansum, np.nansum])
    VARIANCE = ("variance", [da.nanvar, np.nanvar])
    STD_DEV = ("standard_deviation", [da.nanstd, np.nanstd])

    # TODO: deal with all: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ape.html
    UNDEFINED = ("<undefined>", [None])

    def __str__(self) -> str:
        return self.value[0]

    @classmethod
    def _missing_(cls, key):
        for name, member in MethodType.__members__.items():
            if member.value[0] == key:
                return member
        return cls.UNDEFINED

    @property
    def dask_operator(self):
        return self.value[1][0]

    @property
    def numpy_operator(self):
        return self.value[1][-1]


@unique
class RegridMethod(Enum):
    BILINEAR = "bilinear"
    CONSERVATIVE = "conservative"
    CONSERVATIVE_NORMED = "conservative_normed"
    NEAREST_D2S = "nearest_d2s"
    NEAREST_S2D = "nearest_s2d"
    PATCH = "patch"


class LongitudeConvention(Enum):
    POSITIVE_WEST = 1  # 0 to 360
    NEGATIVE_WEST = 2  # -180 to 180


class LatitudeConvention(Enum):
    POSITIVE_TOP = 1  # 90 to -90
    NEGATIVE_TOP = 2  # -90 to 90
