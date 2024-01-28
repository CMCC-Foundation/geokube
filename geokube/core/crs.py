from __future__ import annotations

from pyproj import crs as pyproj_crs

from . import axis
import xarray as xr
import numpy as np


class CRS:
    _DIM_AXES_TYPES: tuple[type[axis.Horizontal], ...] = ()
    _AUX_AXES_TYPES: tuple[type[axis.Horizontal], ...] = ()
    _PYPROJ_TYPE: type[pyproj_crs.CRS] = pyproj_crs.CRS

    __slots__ = ('__dim_axes', '__aux_axes', '__pyproj_crs')

    def __init__(
        self,
        *args,
        dim_axes: tuple[axis.Horizontal, ...] | None = None,
        aux_axes: tuple[axis.Horizontal, ...] | None = None,
        crs: pyproj_crs.CRS | None = None,
        **kwargs,
    ) -> None:
        self.__dim_axes = self._resolve_axes(dim_axes, self._DIM_AXES_TYPES)
        self.__aux_axes = self._resolve_axes(aux_axes, self._AUX_AXES_TYPES)
        self.__pyproj_crs = self._resolve_crs(
            *args, crs=crs, default_type=self._PYPROJ_TYPE, **kwargs
        )

    def _resolve_axes(
        self,
        axes: tuple[axis.Horizontal, ...] | None,
        default_types: tuple[type[axis.Horizontal], ...]
    ) -> tuple[axis.Horizontal, ...]:
        if axes is None:
            return tuple(type_() for type_ in default_types)
        types = tuple(type(axis) for axis in axes)
        if (
            len(types) != len(default_types)
            or set(types) != set(default_types)
        ):
            raise TypeError("'axes' types do not match the expected types")
        return axes

    def _resolve_crs(
        self,
        *args,
        crs: pyproj_crs.CRS | None,
        default_type: type[pyproj_crs.CRS] | None,
        **kwargs
    ) -> pyproj_crs.CRS:
        if crs is None:
            return (
                default_type(*args, **kwargs)
                if default_type is not None else
                None
            )
        if not isinstance(crs, pyproj_crs.CRS):
            raise TypeError(
                "'crs' must be an instance of 'pyproj.crs.crs.CRS' or 'None'"
            )
        if args or kwargs:
            raise ValueError(
                "'args' and 'kwargs' are not accepted when 'crs' is not 'None'"
            )
        return crs

    @property
    def dim_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__dim_axes

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__aux_axes

    @property
    def axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__dim_axes + self.__aux_axes

    @property
    def _crs(self) -> pyproj_crs.CRS:
        return self.__pyproj_crs

    @classmethod
    def from_cf(cls, *args, **kwargs) -> CRS:
        crs = pyproj_crs.CRS.from_cf(*args, **kwargs)

        # HACK: In some cases, the type of `crs` is `pyproj.crs.crs.CRS`, and
        # not any of its subclasses. This is a workaroung to find "the correct"
        # class and create `crs` as an instance of this class.
        # NOTE: This check should be *only* against `CRS` and *not* its
        # subclasses, so:
        # pylint: disable=unidiomatic-typecheck
        if type(crs) is pyproj_crs.crs.CRS:
            crs_kwa = crs.to_json_dict()
            crs_type = getattr(pyproj_crs.crs, crs_kwa['type'])
            crs = crs_type.from_json_dict(crs_kwa)

        match crs:
            case pyproj_crs.GeographicCRS():
                return Geodetic(crs=crs)
            case pyproj_crs.DerivedGeographicCRS():
                return RotatedGeodetic(crs=crs)
            case pyproj_crs.ProjectedCRS():
                # proj_name = crs_kwa['conversion']['method']['name']
                proj_gmn = crs.to_cf()['grid_mapping_name']
                # if (proj_type := _PROJECTION_NAMES.get(proj_name)) is not None:
                if (proj_type := _PROJECTION_GMN.get(proj_gmn)) is not None:
                    proj = object.__new__(proj_type)
                    CRS.__init__(proj, crs=crs)
                    return proj
                else:
                    return Projection(crs=crs)
            case _:
                raise TypeError(f"'crs' type {type(crs)} is not supported")

    def to_cf(self) -> dict:
        return self._crs.to_cf()

    def as_xarray(self) -> xr.DataArray:
        grid_mapping_attrs = self.to_cf()
        grid_mapping_name = grid_mapping_attrs['grid_mapping_name']
        return xr.DataArray(data=np.byte(1),
                            name=grid_mapping_name,
                            attrs=grid_mapping_attrs)


class Geodetic(CRS):
    _DIM_AXES_TYPES = (axis.Latitude, axis.Longitude)
    _AUX_AXES_TYPES = ()
    _PYPROJ_TYPE = pyproj_crs.GeographicCRS

    @property
    def dim_X_axis(self):
        return axis.longitude

    @property
    def dim_Y_axis(self):
        return axis.latitude


class RotatedGeodetic(CRS):
    _DIM_AXES_TYPES = (axis.GridLatitude, axis.GridLongitude)
    _AUX_AXES_TYPES = (axis.Latitude, axis.Longitude)
    _PYPROJ_TYPE = pyproj_crs.DerivedGeographicCRS

    def __init__(
        self,
        grid_north_pole_latitude: float | None = 0.0,
        grid_north_pole_longitude: float | None = 0.0,
        north_pole_grid_longitude: float = 0.0,
        crs: pyproj_crs.DerivedGeographicCRS | None = None,
        dim_axes: tuple[axis.Horizontal, ...] | None = None,
        aux_axes: tuple[axis.Horizontal, ...] | None = None
    ) -> None:
        if crs is None:
            conversion = {
                '$schema':
                    'https://proj.org/schemas/v0.4/projjson.schema.json',
                'type': 'Conversion',
                'name': 'Pole rotation (netCDF CF convention)',
                'method': {
                    'name': 'Pole rotation (netCDF CF convention)'
                },
                'parameters': [
                    {
                        'name':
                            'Grid north pole latitude (netCDF CF convention)',
                        'value': float(grid_north_pole_latitude),
                        'unit': 'degree'
                    },
                    {
                        'name':
                            'Grid north pole longitude (netCDF CF convention)',
                        'value': float(grid_north_pole_longitude),
                        'unit': 'degree'
                    },
                    {
                        'name':
                            'North pole grid longitude (netCDF CF convention)',
                        'value': float(north_pole_grid_longitude),
                        'unit': 'degree'
                    }
                ]
            }
            crs = pyproj_crs.DerivedGeographicCRS(
                base_crs=pyproj_crs.GeographicCRS(), conversion=conversion
            )
        super().__init__(dim_axes=dim_axes, aux_axes=aux_axes, crs=crs)

    @property
    def dim_X_axis(self) -> axis.Axis:
        return axis.grid_longitude

    @property
    def dim_Y_axis(self) -> axis.Axis:
        return axis.grid_latitude


class Projection(CRS):
    _DIM_AXES_TYPES = (axis.Y, axis.X)
    _AUX_AXES_TYPES = (axis.Latitude, axis.Longitude)
    _PYPROJ_TYPE = pyproj_crs.ProjectedCRS

    # def __init__(self, conversion):
    #     pass

    @property
    def dim_X_axis(self) -> axis.Axis:
        return axis.x

    @property
    def dim_Y_axis(self) -> axis.Axis:
        return axis.y


class UnknownProjection(Projection):
    _PYPROJ_TYPE = None


# NOTE: The classes `TransverseMercatorConversion` and
# `LambertConformalConic1SPConversion` have identical signatures for the
# constructors and thus the identical implementations.
class TransverseMercatorProjection(Projection):
    def __init__(
        self,
        latitude_natural_origin: float = 0.0,
        longitude_natural_origin: float = 0.0,
        false_easting: float = 0.0,
        false_northing: float = 0.0,
        scale_factor_natural_origin: float = 1.0,
        dim_axes: tuple[axis.Horizontal, ...] | None = None,
        aux_axes: tuple[axis.Horizontal, ...] | None = None,
    ) -> None:
        oper = pyproj_crs.coordinate_operation.TransverseMercatorConversion(
            latitude_natural_origin=latitude_natural_origin,
            longitude_natural_origin=longitude_natural_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor_natural_origin=scale_factor_natural_origin
        )
        crs = self._PYPROJ_TYPE(conversion=oper.to_json_dict())
        super().__init__(dim_axes=dim_axes, aux_axes=aux_axes, crs=crs)


class LambertConformalConic1SPProjection(Projection):
    def __init__(
        self,
        latitude_natural_origin: float = 0.0,
        longitude_natural_origin: float = 0.0,
        false_easting: float = 0.0,
        false_northing: float = 0.0,
        scale_factor_natural_origin: float = 1.0,
        dim_axes: tuple[axis.Horizontal, ...] | None = None,
        aux_axes: tuple[axis.Horizontal, ...] | None = None,
    ) -> None:
        oper = (
            pyproj_crs.coordinate_operation.LambertConformalConic1SPConversion(
                latitude_natural_origin=latitude_natural_origin,
                longitude_natural_origin=longitude_natural_origin,
                false_easting=false_easting,
                false_northing=false_northing,
                scale_factor_natural_origin=scale_factor_natural_origin
            )
        )
        crs = self._PYPROJ_TYPE(conversion=oper.to_json_dict())
        super().__init__(dim_axes=dim_axes, aux_axes=aux_axes, crs=crs)


_PROJECTION_NAMES = {
    'Lambert Conic Conformal (1SP)': LambertConformalConic1SPProjection,
    'Transverse Mercator': TransverseMercatorProjection
}

_PROJECTION_GMN = {
    'lambert_conformal_conic': LambertConformalConic1SPProjection,
    'transverse_mercator': TransverseMercatorProjection
}
