"""
Axis
====

A domain axis construct defined by a dimension or scalar coordinate
variable.  The data array that belongs to a field spans the axis
constructs of the domain.

:obj:`geokube` axes are defined as hashable objects suitable to be the
keys of dictionaries and members of sets.

Functions
---------

custom(type_name[, units, encoding, dtype])
    Create a user-defined axis subclass and return its instance.

Classes
-------

Axis
    Base class for axis constructs.

Spatial
    Base class for spatial axis constructs.

Horizontal
    Base class for horizontal spatial axis constructs.

Longitude
    Longitude axis.

Latitude
    Latitude axis.

GridLongitude
    Grid longitude axis along the Y dimension.

GridLatitude
    Grid latitude axis along the X dimension.

X
    General X horizontal spatial axis.

Y
    General Y horizontal spatial axis.

Elevation
    Base class for vertical, i.e. elevation spatial axis constructs.

Z
    General Z elevation, i.e. vertical spatial axis.

Vertical
    Vertical axis.

Height
    Height axis.

Depth
    Depth axis.

Time
    Time axis.

UserDefined
    Base class for custom user-defined axis constructs.

"""

from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pint

from .units import units


# pylint: disable=unused-argument


class Axis(str):
    """
    Base class for axis constructs.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    # HACK: `Axis` inherits `str` to be capable of acting as a dimension in
    # `xarray` data structures. If `xarray` becomes capable of handling all
    # hashables as dimensions, the inheritance from `str` will not be
    # necessary any more. In that case, `Axis` might inherit from `Hashable`.

    _DEFAULT_UNITS_: pint.Unit
    _DEFAULT_ENCODING_: { 'dtype': np.dtype }

    __slots__ = ('__units', '__encoding')

    def __new__(
        cls,
        units: pint.Unit | str | None = None,
        encoding: Mapping[str, Any] | None = None
    ):
        return str.__new__(cls, cls.__name__)

    def __init__(
        self,
        units: pint.Unit | str | None = None,
        encoding: Mapping[str, Any] | None = None
    ) -> None:

        self.__units = pint.Unit(units or self._DEFAULT_UNITS_)
        self.__encoding = self._DEFAULT_ENCODING_ if encoding is None else dict(encoding)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(units='{self.__units}', encoding={self.__encoding})"
        )

    @property
    def name(self) -> str:
        return str(self)

    @property
    def units(self) -> pint.Unit:
        return self.__units

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.__encoding['dtype'])

    @property
    def encoding(self) -> dict[str, Any]:
        return self.__encoding


class Spatial(Axis):
    """
    Base class for spatial axis constructs.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_DTYPE_ = 'float64'


class Horizontal(Spatial):
    """
    Base class for horizontal spatial axis constructs.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    pass


class Longitude(Horizontal):
    """
    Longitude axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['degrees_east']
    _DEFAULT_ENCODING_ = { 'standard_name': 'longitude',
                           'dtype': 'float64'}


class Latitude(Horizontal):
    """
    Latitude axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['degrees_north']
    _DEFAULT_ENCODING_ = { 'standard_name': 'latitude',
                           'dtype': 'float64'}


class GridLongitude(Horizontal):
    """
    Grid longitude axis along the Y dimension.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['degrees']
    _DEFAULT_ENCODING_ = { 'standard_name': 'grid_longitude',
                           'dtype': 'float64'}


class GridLatitude(Horizontal):
    """
    Grid latitude axis along the Y dimension.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['degrees']
    _DEFAULT_ENCODING_ = { 'standard_name': 'grid_latitude',
                           'dtype': 'float64'}


class X(Horizontal):
    """
    General X horizontal spatial axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'axis': 'X',
                           'dtype': 'float64'}


class Y(Horizontal):
    """
    General Y horizontal spatial axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'axis': 'Y',
                           'dtype': 'float64'}


class Elevation(Spatial):
    """
    Base class for vertical, i.e. elevation spatial axis constructs.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']


class Z(Elevation):
    """
    General Z elevation, i.e. vertical spatial axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'axis': 'Z',
                           'dtype': 'float64'}


class Vertical(Elevation):
    """
    Vertical axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'standard_name': 'height',
                           'positive': 'up',
                           'dtype': 'float64'}


class Height(Elevation):
    """
    Height axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'standard_name': 'height',
                           'positive': 'up',
                           'dtype': 'float64'}


class Depth(Elevation):
    """
    Depth axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'standard_name': 'depth',
                           'positive': 'down',
                           'dtype': 'float64'}


class Time(Axis):
    """
    Time axis.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    """

    _DEFAULT_UNITS_ = units['']
    _DEFAULT_ENCODING_ = { 'standard_name': 'time',
                           'axis': 'T',
                           'dtype': 'datetime64'}


class UserDefined(Axis):
    """
    Base class for custom user-defined axis constructs.

    Parameters
    ----------
    units : pint unit or string, default: None
        The units associated to the axis.
    encoding : dict_like, default: None
        The encoding associated to the axis.

    Attributes
    ----------
    name : str
        The name of the axis.
    units : pint units
        The units associated to the axis.
    dtype : NumPy data-type
        Data-type of the data array associated to the axis.
    encoding : dict
        The encoding associated to the axis.

    See Also
    --------
    custom :
        Create a user-defined axis subclass and return its instance.

    """

    # FIXME: Hash cannot be the class name -> redefine hash with axis name.
    _DEFAULT_UNITS_ = units['']
    _DEFAULT_ENCODING_ = { 'dtype': 'str'}

x = X()
y = Y()
z = Z()
grid_latitude = GridLatitude()
grid_longitude = GridLongitude()
latitude = Latitude()
longitude = Longitude()
vertical = Vertical()
height = Height()
depth = Depth()
time = Time()
timedelta = Time(encoding={'dtype': 'timedelta64'})


def create(
    name: str, units: pint.Unit | str, encoding: dict
) -> None:
    # TODO: Implement this.
    pass


__predefined_axis__ = {
    'x': x,
    'y': y,
    'z': z,
    'grid_latitude': grid_latitude,
    'grid_longitude': grid_longitude,
    'latitude': latitude,
    'longitude': longitude,
    'vertical': vertical,
    'height': height,
    'depth': depth,
    'time': time,
    'timedelta': timedelta
}


def _from_string(name: str):
    return __predefined_axis__.get(name)


UserDefinedT_co = TypeVar(
    'UserDefinedT_co', bound=UserDefined, covariant=True
)


def custom(
    name: str,
    units: str | pint.Unit | None = None,
    encoding: Mapping[str, Any] | None = None,
) -> UserDefinedT_co:
    """
    Create a user-defined axis subclass and return its instance.

    This is a utility function that helps creating instances of
    user-defined axes, e.g. discrete axis.  It creates a new subclass of
    the class :class:`UserDefined` with the provided default `units`,
    `encoding`, and `dtype`; instantiates this class; and returns the
    instance.

    Parameters
    ----------
    name : str
        The name of the user-defined class.
    units : pint unit or str, default: ''
        Default units of the user-defined class.
    encoding : dict_like, default: None
        Default encoding of the user-defined class.

    Returns
    -------
    obj : UserDefined
        User defined axis.

    See Also
    --------
    UserDefined : Base class for custom user-defined axis constructs.

    Examples
    --------
    >>> discrete = axis.custom(
    ...     name='Discrete',
    ...     units='m',
    ...     encoding={'standard_name': 'name', 'dtype': 'float64'}
    ... )
    >>> discrete
    Discrete(units='meter', encoding={'standard_name': 'name', 'dtype': 'float64'})

    """
    # NOTE: To be used in this manner: `ensemble = custom('Ensemble')`.
    name = str(name)
    base = (UserDefined,)
    dict_ = {
        '_DEFAULT_UNITS_': pint.Unit(units or UserDefined._DEFAULT_UNITS_),
        '_DEFAULT_ENCODING_': dict(encoding or UserDefined._DEFAULT_ENCODING_),
    }
    cls_ = type(name, base, dict_)
    obj = cls_()
    return obj
