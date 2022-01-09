from typing import Optional
import warnings

import cf_units as cf
import xarray as xr

from geokube.core.axis import Axis, AxisType
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger

UNKNOWN_UNIT = cf.Unit(None)


class Dimension:

    _LOG = HCubeLogger(name="Dimension")

    def __init__(
        self,
        name: str,
        axis: Axis,
    ):
        self._name = name
        self._axis = axis

    @property
    def name(self) -> str:
        return self._name

    @property
    def axis(self):
        return self._axis

    @property
    def atype(self) -> str:
        return self._axis.atype

    @property
    def default_units(self) -> cf.Unit:
        return self._axis.atype.default_units

    @classmethod
    @log_func_debug
    def from_xarray_dataarray(cls, da: xr.DataArray):
        axis = Axis.from_xarray_dataarray(da)
        attrs = da.attrs
        # Attrs['Axis'] -> Attrs['standard_name'] -> da.name
        name = attrs.get("standard_name", da.attrs.get("axis", da.name))
        return Dimension(name=name, axis=axis)

    @classmethod
    def _parse_units(cls, name: str, calendar: Optional[str] = None) -> cf.Unit:
        # TODO: more logic
        try:
            return cf.Unit(name, calendar=calendar)
        except ValueError:
            warnings.warn(
                f"Failed to create CF unit for values: `{name}`. Using <unknown>!"
            )
            cls._LOG.warn(
                f"Failed to create CF unit for values: `{name}`. Using <unknown>!"
            )
            # Some units are not valid like in case of:
            # /data/inputs/ERA5/single-levels/reanalysis/1979/era5_single_levels_reanalysis_mean_wave_direction_0.25x0.25_1979.nc
            # where units is `Degree true`
            return UNKNOWN_UNIT

    def __eq__(self, other):
        if isinstance(other, Dimension):
            return (self.name == other.name) and (self._type == other._type)
        else:
            False

    def __ne__(self, other):
        return not (self == other)
