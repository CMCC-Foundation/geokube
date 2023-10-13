from pyproj import crs as pyproj_crs


class CRS(pyproj_crs.CRS):
    __slots__ = ('__dim_axes', '__aux_axes', '__pyproj_crs')

    _DIM_AXES_TYPES: tuple[type[axis.Horizontal], ...]
    _AUX_AXES_TYPES: tuple[type[axis.Horizontal], ...]
    _PYPROJ_TYPE: type[pyproj_crs.CRS]

    def __init__(
        self,
        *args,
        dim_axes: tuple[axis.Horizontal, ...] | None = None,
        aux_axes: tuple[axis.Horizontal, ...] | None = None,
        **kwargs,
    ) -> None:
        # TODO: Make this better.
        self.__pyproj_crs = (
            kwargs['crs']
            if 'crs' in kwargs else
            self._PYPROJ_TYPE(*args, **kwargs)
        )

        # TODO: Move this to a separate internal method.
        if dim_axes is None:
            self.__dim_axes = tuple(
                axis_type() for axis_type in self._DIM_AXES_TYPES
            )
        else:
            dim_axes_types = [type(dim_axis) for dim_axis in dim_axes]
            if (
                len(dim_axes_types) != len(self._DIM_AXES_TYPES)
                or set(dim_axes_types) != set(self._DIM_AXES_TYPES)
            ):
                raise TypeError(
                    "'dim_axes' types do not matched the expected types"
                )
            self.__dim_axes = dim_axes

        # TODO: Move this to a separate internal method.
        if aux_axes is None:
            self.__aux_axes = tuple(
                axis_type() for axis_type in self._AUX_AXES_TYPES
            )
        else:
            aux_axes_types = [type(aux_axis) for aux_axis in aux_axes]
            if (
                len(aux_axes_types) != len(self._AUX_AXES_TYPES)
                or set(aux_axes_types) != set(self._AUX_AXES_TYPES)
            ):
                raise TypeError(
                    "'aux_axes' types do not matched the expected types"
                )
            self.__aux_axes = aux_axes

    @property
    def dim_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__dim_axes

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__aux_axes

    @property
    def axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__dim_axes + self.__aux_axes

    @classmethod
    def from_cf(*args, **kwargs) -> CRS:
        # TODO: Finish this.
        return type(self)(crs=)
        


# class RotatedGeodetic(DerivedGeographicCRS):
#     _DIM_AXES_TYPES = (axis.GridLatitude, axis.GridLongitude)
#     _AUX_AXES_TYPES = (axis.Latitude, axis.Longitude)

#     def __init__(self, *args, **kwargs, dim_axes=None, aux_axes=None):
#         super().__init__(*args, **kwargs)
#         if dim_axes is None:
#             self.__dim_axes = tuple(axis_type() for axis_type in _DIM_AXES_TYPES)
#         else:
#             # TODO: Control the classes.
#             self.__dim_axes = dim_axes


# class CRS -> wrapper of pyproj.crs.CRS


# class RotatedGeodetic(CRS):
#     _DIM_AXES_TYPES = (axis.GridLatitude, axis.GridLongitude)
#     _AUX_AXES_TYPES = (axis.Latitude, axis.Longitude)

#     __slots__ = ('__dim_axes', '__aux_axes', '__pyproj_crs')

#     def __init__(self, *args, **kwargs, dim_axes=None, aux_axes=None):
#         self.__pyproj_crs = DerivedGeographicCRS(*args, **kwargs)
#         if dim_axes is None:
#             self.__dim_axes = tuple(axis_type() for axis_type in _DIM_AXES_TYPES)
#         else:
#             # TODO: Control the classes.
#             self.__dim_axes = dim_axes

#     @classmethod
#     def from_cf(cls, *args, **kwargs):
#         return type(self)(*args, **kwargs)

#     @property
#     def dim_axes(self):
#         return self.__dim_axes


# CRS.dim_axes[0].encoding.update(...)

# axis.encoding -> dict
# axis.__hash__ -> acc. to the class name
# New axis types like axis.GridLatitude, axis.GridLongitude, axis.X, axis.Y (derived from axis.Horizontal)
# Height, Depth, Vertical, Z (derived from Elevation)
# Keep default objects

# from cartopy.crs import CRS, Geodetic, Projection, RotatedGeodetic

# from . import axis


# CRS.AXES: tuple[axis.Horizontal, ...] = ()
# Geodetic.AXES = (axis.latitude, axis.longitude)
# RotatedGeodetic.AXES = (
#     axis.grid_latitude, axis.grid_longitude, axis.latitude, axis.longitude
# )
# Projection.AXES = (axis.y, axis.x, axis.latitude, axis.longitude)


# try:
#     from cartopy import crs

#     from . import axis

#     class CRS:
#         AXES: tuple[axis.Horizontal, ...] = ()


#     class Geodetic(CRS, crs.Geodetic):
#         AXES = (axis.latitude, axis.longitude)


#     class RotatedGeodetic(CRS, crs.RotatedGeodetic):
#         AXES = (axis.grid_latitude, axis.grid_longitude, *Geodetic.AXES)


#     class Projection(CRS, crs.Projection):
#         AXES = (axis.y, axis.x, *Geodetic.AXES)
# except ImportError:
#     from . import axis
    
#     class CRS:
#         AXES: tuple[axis.Horizontal, ...] = ()

#         def transform_points(self, *args, **kwargs):
#             raise NotImplementedError(
#                 "f'{self.__class__.__name__}.transform_points' can work only "
#                 "if 'cartopy' is installed"
#             )


#     class Geodetic(CRS):
#         AXES = (axis.latitude, axis.longitude)


#     class RotatedGeodetic(CRS):
#         AXES = (axis.grid_latitude, axis.grid_longitude, *Geodetic.AXES)


#     class Projection(CRS):
#         AXES = (axis.y, axis.x, *Geodetic.AXES)
