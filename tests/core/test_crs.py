import pytest

import geokube.core.axis as axis
from geokube.core.crs import Geodetic, RotatedGeodetic, TransverseMercatorProjection

def test_geodetic_crs():
    crs = Geodetic()
    assert crs.axes == (axis.latitude, axis.longitude)
    assert crs.dim_axes == (axis.latitude, axis.longitude)
    assert crs.aux_axes == ()
    assert crs.dim_X_axis == axis.longitude
    assert crs.dim_Y_axis == axis.latitude

def test_rotated_crs():
    crs = RotatedGeodetic()
    assert crs.dim_axes == (axis.grid_latitude, axis.grid_longitude)
    assert crs.aux_axes == (axis.latitude, axis.longitude)
    assert crs.dim_X_axis == axis.grid_longitude
    assert crs.dim_Y_axis == axis.grid_latitude

def test_projection_crs():
    crs = TransverseMercatorProjection()
    assert crs.dim_axes == (axis.y, axis.x)
    assert crs.aux_axes == (axis.latitude, axis.longitude)
    assert crs.dim_X_axis == axis.x
    assert crs.dim_Y_axis == axis.y