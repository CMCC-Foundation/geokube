from __future__ import annotations
from numbers import Number
import dask.array as da
import pkg_resources
from functools import partial
import numpy as np
import re
import pint

def maybe_adjust_object_by_unit(
    magnitude, expected_units=None
) -> Number | np.ndarray | da.Array:
    res = None
    if hasattr(magnitude, "data"):
        data_to_convert = magnitude.data
    elif hasattr(magnitude, "magnitude"):
        data_to_convert = magnitude.magnitude
    elif isinstance(magnitude, (Number, np.ndarray, da.Array)):
        data_to_convert = magnitude
        return data_to_convert, None, None
    if expected_units is None:
        return data_to_convert, None, None
    if hasattr(magnitude, "units"):
        source_units = magnitude.units
        if isinstance(source_units, TimeReference):
            # TODO: logic for time conversion should be added in the future
            return data_to_convert, source_units, source_units
    else:
        raise ValueError(
            f"`magnitude` parameter does not contain units and it's not of"
            f" type [`Number`, `numpy.ndarray`, `da.Array`]!"
        )
    if isinstance(expected_units, str):
        target_units = units(expected_units)
    elif hasattr(expected_units, "units"):
        target_units = expected_units.units
    elif isinstance(expected_units, (Number, np.ndarray, da.Array)):
        target_units = source_units
        return data_to_convert, source_units, target_units
    assert isinstance(source_units, pint.Unit), (
        "`source_units` object should be of `pint.Unit` class not"
        f" `{type(source_units).__name__}`"
    )
    try:
        res = pint.Quantity(data_to_convert, source_units).to(target_units).m
    except pint.errors.DimensionalityError as e:
        # _LOG.error(
        #     f"Failed to convert units `{source_units}` to `{target_units}`"
        # )
        raise ValueError(
            f"Units `{source_units}` cannot be converted to `{target_units}`!"
        ) from e
    return res, source_units, target_units


class CFUnitRegistry(pint.UnitRegistry):
    def parse_expression(
        self,
        input_string: str,
        case_sensitive: None | bool = None,
        use_decimal: bool = False,
        calendar: None | str = None,
        **values,
    ) -> pint.Unit | TimeReference:
        if isinstance(input_string, str) and (" since " in input_string):
            cf_time = cf.Unit(input_string, calendar=calendar)
            return TimeReference(
                cf_time.cftime_unit, calendar=cf_time.calendar
            )
        return (
            super()
            .parse_expression(
                input_string=input_string,
                case_sensitive=case_sensitive,
                use_decimal=use_decimal,
                **values,
            )
            .u
        )

    __call__ = parse_expression


class TimeReference:
    def __init__(
        self,
        time_origin: str,
        calendar: None | str = "gregorian",
    ):
        self._calendar = self._time_origin = None
        self._calendar = calendar
        self._time_origin = time_origin

    @property
    def time_origin(self):
        return self._time_origin

    @property
    def calendar(self):
        return self._calendar

    @property
    def dimensionless(self):
        return True

    def __eq__(self, other):
        if isinstance(other, TimeReference):
            return (self._calendar == other._calendar) and (
                self._time_origin == other._time_origin
            )
        return False

    def __ne__(self, other):
        return not (self == other)

    def to(value, time_reference):
        raise NotImplementedError()

    def __str__(self):
        return self._time_origin


units = CFUnitRegistry(
    autoconvert_offset_to_baseunit=True,
    preprocessors=[
        partial(
            re.sub,
            r"(?<=[A-Za-z\)])(?![A-Za-z\)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])",
            "**",
        ),
        lambda string: string.replace("%", "percent"),
        lambda string: string.replace(
            r"(0 - 1)", "percent"
        ),  # ERA5-Reanalysis `tcc` field
        lambda string: string.replace(
            r"Degree true", "degree"
        ),  # ERA5-Reanalysis `mwd` field
        lambda string: string.replace(
            r"m of water equivalent", "m"
        ),  # ERA5-Reanalysis `sf` field
    ],
)
units.load_definitions(
    pkg_resources.resource_filename("geokube", "static/units/units_def.txt")
)
units.setup_matplotlib()
pint.set_application_registry(units)
