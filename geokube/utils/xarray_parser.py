import logging
import re
from itertools import chain
from string import Formatter, Template
from typing import Iterable, Mapping, Optional, Tuple, Union
import warnings
import numpy as np
import xarray as xr

import geokube.core.coord_system as crs

BOUNDS_PATTERN = re.compile(r".*(bnds|bounds).*$", re.IGNORECASE)

logger = logging.getLogger(__name__)


def form_id(id_pattern, attrs):
    fmt = Formatter()
    _, field_names, _, _ = zip(*fmt.parse(id_pattern))
    field_names = [f for f in field_names if f]
    # Replace intake-like placeholder to string.Template-like ones
    present_values = {}
    for k in field_names:
        if k not in attrs:
            warnings.warn(
                f"Requested id component - `{k}` is not present among provided attributes!"
            )
            # TODO: what if there is missing attibute, for instance for a dimension? Now pattern is created with <undefined>
            present_values[k] = "<unknown>"
        else:
            present_values[k] = attrs[k]
        id_pattern = id_pattern.replace(
            f"{{{k}}}", f"${{{k}}}"
        )  # "{some_field}" -> "${some_field}"
    template = Template(id_pattern)
    return template.substitute(**present_values)
