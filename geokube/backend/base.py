import datetime
import os
import re
from typing import Any, Mapping, NoReturn, Optional, Tuple

import numpy as np
import pandas as pd


def _rev_format_to_glob_pattern(val: str):
    return os.sep.join(
        map(lambda s: re.sub("{.*}", "*", s), val.split(os.sep))
    )


def convert_cftimes_to_numpy(obj):
    for key in obj.coords:
        if obj[key].dtype == np.dtype("O"):
            try:
                obj[key] = obj.indexes[key].to_datetimeindex()
            except AttributeError:
                pass
    return obj
