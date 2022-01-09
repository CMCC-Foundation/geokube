from enum import Enum
from typing import List, Mapping, Tuple


class CFAttributes(Enum):

    NETCDF_NAME = "netcdf_name"

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
    def split_attrs(
        cls, attrs: Mapping[str, str]
    ) -> Tuple[Mapping[str, str], Mapping[str, str]]:
        properties = attrs.copy()
        cf_encoding = {k: properties.pop(k) for k in cls.get_names() if k in attrs}
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
    CFAttributes.UNITS.value,
    CFAttributes.CALENDAR.value,
    CFAttributes.MISSING_VALUE.value,
    CFAttributes.FILL_VALUE.value,
    CFAttributes.SCALE_FACTOR.value,
    CFAttributes.ADD_OFFSET.value,
)


def split_to_attrs_and_encoding(
    mapping: Mapping[str, str]
) -> Tuple[Mapping[str, str], Mapping[str, str]]:
    attrs, encoding = {}, {}
    for k, v in mapping.items():
        if k in ENCODING_PROP:
            encoding[k] = v
        else:
            attrs[k] = v
    return (attrs, encoding)
