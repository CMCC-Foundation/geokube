import re
from enum import Enum
from typing import List, Mapping, Tuple


class CFAttributes(Enum):
    # encoding for names and dims
    NETCDF_NAME = "name"
    NETCDF_DIMS = "dims"

    # Description of data:
    UNITS = "units"
    STANDARD_NAME = "standard_name"
    LONG_NAME = "long_name"
    ANCILLARY_VARIABLES = " ancillary_variables"
    MISSING_VALUE = "missing_value"
    VALID_RANGE = "valid_range"
    VALID_MIN = "valid_min"
    VALID_MAX = "valid_max"
    FLAG_VALUES = "flag_values"
    FLAG_MEANINGS = "flag_meanings"
    FLAG_MASKS = "flag_masks"
    FILL_VALUE = "_FillValue"

    # Coordinate systems
    COORDINATES = "coordinates"
    AXIS = "axis"
    BOUNDS = "bounds"
    GRID_MAPPING = "grid_mapping"
    FORMULA_TERMS = "formula_terms"
    CALENDAR = "calendar"
    POSITIVE = "positive"

    # Data packing
    ADD_OFFSET = "add_offset"
    SCALE_FACTOR = "scale_factor"
    COMPRESS = "compress"

    # Data cell properties and methods
    CELL_MEASURES = "cell_measures"
    CELL_METHODS = "cell_methods"
    CLIMATOLOGY = "climatology"

    @classmethod
    def get_names(cls) -> List[str]:
        return [a.value for a in cls]

    @classmethod
    def split_to_props_encoding(
        cls, attrs: Mapping[str, str]
    ) -> Tuple[Mapping[str, str], Mapping[str, str]]:
        properties = attrs.copy()
        cf_encoding = {
            k: properties.pop(k) for k in cls.get_names() if k in attrs
        }
        return (properties, cf_encoding)


ENCODING_PROP = (
    "source",
    "dtype",
    "original_shape",
    "chunksizes",
    "zlib",
    "shuffle",
    "complevel",
    "fletcher32",
    "contiguous",
    CFAttributes.COORDINATES.value,
    CFAttributes.CALENDAR.value,
    CFAttributes.GRID_MAPPING.value,
    CFAttributes.MISSING_VALUE.value,
    CFAttributes.FILL_VALUE.value,
    CFAttributes.SCALE_FACTOR.value,
    CFAttributes.ADD_OFFSET.value,
)


def is_time_unit(unit):
    return "since" in unit if isinstance(unit, str) else False


_TIME_REF_RE = re.compile(r"^\s*(\w+)\s+since\s+(.+)$", re.IGNORECASE)

# Reference-unit steps xarray/cftime CANNOT decode for a real-world calendar: months and
# years have variable length, so cftime only supports them for the ``360_day`` calendar.
# Every other step (day/hour/minute/second/...) decodes natively.
_UNDECODABLE_TIME_STEPS = frozenset({"month", "year"})


def parse_time_reference(unit):
    """Split a CF reference time unit into ``(step, reference)``.

    ``"months since 1993-01-01 00:00:00"`` -> ``("month", "1993-01-01 00:00:00")``.
    The step is lower-cased with a trailing ``s`` stripped (``months`` -> ``month``).
    Returns ``None`` if ``unit`` is not a ``"<step> since <date>"`` string.
    """
    if not isinstance(unit, str):
        return None
    match = _TIME_REF_RE.match(unit)
    if match is None:
        return None
    step = match.group(1).lower()
    if step.endswith("s"):
        step = step[:-1]
    return step, match.group(2).strip()


def is_undecodable_time_unit(unit, calendar=None):
    """True for a ``"months since ..."`` / ``"years since ..."`` reference unit that
    xarray/cftime cannot decode (any calendar except ``360_day``).

    These are the only time units geokube must decode itself; ``day``/``hour``/``minute``/
    ``second`` "since" units (and the ``AxisType.TIME`` default ``"hours since 1970-01-01"``)
    decode natively via xarray and return ``False`` here.
    """
    parsed = parse_time_reference(unit)
    if parsed is None:
        return False
    step, _ = parsed
    if step not in _UNDECODABLE_TIME_STEPS:
        return False
    cal = calendar.lower() if isinstance(calendar, str) else ""
    return cal != "360_day"


def in_encoding(key, unit=None):
    return is_time_unit(unit) or key in ENCODING_PROP


def split_to_xr_attrs_and_encoding(
    mapping: Mapping[str, str]
) -> Tuple[Mapping[str, str], Mapping[str, str]]:
    attrs, encoding = {}, {}
    if mapping is not None:
        for k, v in mapping.items():
            if in_encoding(k, v):
                encoding[k] = v
            else:
                attrs[k] = v
    return (attrs, encoding)
