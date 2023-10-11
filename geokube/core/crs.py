from pyproj.crs import CRS, DerivedGeographicCRS, GeographicCRS, ProjectedCRS

from . import axis


CRS.AXES: tuple[axis.Horizontal, ...] = ()
GeographicCRS.AXES = (axis.latitude, axis.longitude)
DerivedGeographicCRS.AXES = (
    axis.grid_latitude, axis.grid_longitude, axis.latitude, axis.longitude
)
ProjectedCRS.AXES = (axis.y, axis.x, axis.latitude, axis.longitude)


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
