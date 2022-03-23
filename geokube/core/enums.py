import re
from contextlib import suppress
from enum import Enum, unique
from typing import Optional, Union

import dask.array as da
import numpy as np


class MethodType(Enum):
    # From https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ape.html
    POINT = ("point", "point", [None])
    MAX = ("max", "maximum", [da.nanmax, np.nanmax])
    MIN = ("min", "minimum", [da.nanmin, np.nanmin])
    MEAN = ("mean", "mean", [da.nanmean, np.nanmean])
    SUM = ("sum", "sum", [da.nansum, np.nansum])
    VARIANCE = ("var", "variance", [da.nanvar, np.nanvar])
    STD_DEV = ("std", "standard_deviation", [da.nanstd, np.nanstd])

    # TODO: deal with all: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ape.html
    UNDEFINED = ("<undefined>", "<undefined>", [None])

    def __str__(self) -> str:
        return self.value[1]

    @classmethod
    def _missing_(cls, key):
        for _, member in MethodType.__members__.items():
            # NOTE: we check either short name or operator label
            # Label is required to handle cell_methods properly
            if (member.value[0] == key) or (member.value[1] == key):
                return member
        return cls.UNDEFINED

    @property
    def dask_operator(self):
        return self.value[2][0]

    @property
    def numpy_operator(self):
        return self.value[2][-1]


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
