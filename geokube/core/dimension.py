from typing import Optional
import warnings

from geokube.core.unit import Unit
import xarray as xr

from geokube.core.axis import Axis, AxisType
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger


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
    def default_units(self) -> Unit:
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
    def _parse_units(cls, name: str, calendar: Optional[str] = None) -> Unit:
        return Unit(name, calendar=calendar)

    def __eq__(self, other):
        if isinstance(other, Dimension):
            return (self.name == other.name) and (self._type == other._type)
        else:
            False

    def __ne__(self, other):
        return not (self == other)
