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
