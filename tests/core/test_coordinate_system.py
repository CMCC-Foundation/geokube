import pytest

import geokube.core.axis as axis
from geokube.core.coord_system import SpatialCoordinateSystem, CoordinateSystem
from geokube.core.crs import Geodetic, RotatedGeodetic


def test_spatial_coordinate_system():
    cs = SpatialCoordinateSystem(crs=Geodetic(), elevation=axis.vertical)

    assert cs.axes == (axis.vertical, axis.latitude, axis.longitude)
    assert cs.dim_axes == (axis.vertical, axis.latitude, axis.longitude)
    assert cs.aux_axes == ()
    assert isinstance(cs.crs, Geodetic)

    cs = SpatialCoordinateSystem(
        crs=RotatedGeodetic(), elevation=axis.vertical
    )

    assert cs.dim_axes == (
        axis.vertical,
        axis.grid_latitude,
        axis.grid_longitude,
    )
    assert cs.aux_axes == (axis.latitude, axis.longitude)
    assert isinstance(cs.crs, RotatedGeodetic)


def test_coordinate_system():

    cs = CoordinateSystem(
        horizontal=RotatedGeodetic(),
        elevation=axis.vertical,
        time=axis.time,
        user_axes=[axis.custom("ensemble")],
    )

    assert cs.dim_axes == (
        axis.custom("ensemble"),
        axis.time,
        axis.vertical,
        axis.grid_latitude,
        axis.grid_longitude,
    )
    assert cs.aux_axes == (axis.latitude, axis.longitude)
    assert cs.time == axis.time
    assert cs.elevation == axis.vertical
