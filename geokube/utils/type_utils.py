from collections.abc import Iterable
from enum import Enum
from numbers import Number
from typing import Any, Hashable, Mapping, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr


class Types(Enum):
    INT = "int"
    STR = "str"
    BOOL = "bool"
    EMPTY = "empty"
    UNKNOWN = "unknown"


BoolTypes = (bool, np.bool_)
IntTypes = (int, np.int8, np.int16, np.int32, np.int64)
CollectionTypes = Iterable


def infer_type(key: Any) -> Types:
    if isinstance(key, slice) and (
        isinstance(key.start, IntTypes) or isinstance(key.stop, IntTypes)
    ):
        return Types.INT
    if isinstance(key, CollectionTypes) and not isinstance(key, str):
        try:
            f = next(iter(key))
            if isinstance(f, BoolTypes):
                return Types.BOOL
            if isinstance(f, IntTypes):
                return Types.INT
            if isinstance(f, str):
                return Types.STR
            return Types.UNKNOWN
        except StopIteration:
            return Types.EMPTY
    if isinstance(key, BoolTypes):
        return Types.BOOL
    if isinstance(key, IntTypes):
        return Types.INT
    if isinstance(key, str):
        return Types.STR
    return Types.UNKNOWN
