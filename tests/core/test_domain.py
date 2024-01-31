import pytest

import numpy as np
import pint
import xarray as xr

import geokube.core.axis as axis
from geokube.core.domain import Points, Profiles, Grid
from geokube.core.crs import Geodetic
from geokube.core.coord_system import CoordinateSystem
from geokube.core.units import units

from tests.fixtures.features import *

#
# Points Tests
#


def test_create_points_1(points1):
    assert points1.dim_axes == (
        axis.time,
        axis.vertical,
        axis.latitude,
        axis.longitude,
    )
    assert points1.number_of_points == 2

    assert np.all(
        points1.coords[axis.vertical]
        == pint.Quantity([10.5, 11.2], units["meter"])
    )
    assert np.all(
        points1.coords[axis.latitude]
        == pint.Quantity([42.2, 56.2], units["degrees_north"])
    )
    assert np.all(
        points1.coords[axis.longitude]
        == pint.Quantity([-12.2, 10.2], units["degrees_east"])
    )


def test_create_points_2(points2):
    assert points2.dim_axes == (
        axis.time,
        axis.vertical,
        axis.latitude,
        axis.longitude,
    )
    assert points2.number_of_points == 2

    assert np.all(
        points2.coords[axis.vertical]
        == pint.Quantity([10.5, 11.2], units["meter"])
    )
    assert np.all(
        points2.coords[axis.latitude]
        == pint.Quantity([42.2, 56.2], units["degrees_north"])
    )
    assert np.all(
        points2.coords[axis.longitude]
        == pint.Quantity([-12.2, 10.2], units["degrees_east"])
    )


def test_bounding_box_points(points1, bbox_all, bbox_partial, bbox_noone):
    pts_bb = points1.bounding_box(**bbox_all)
    assert pts_bb.number_of_points == 2

    pts_bb = points1.bounding_box(**bbox_partial)
    assert pts_bb.number_of_points == 1
    assert pts_bb.coords[axis.vertical] == pint.Quantity(10.5, units["meter"])

    pts_bb = points1.bounding_box(**bbox_noone)
    assert pts_bb.number_of_points == 0


def test_nearest_horizontal_points(points1):
    pts = points1.nearest_horizontal(latitude=41, longitude=-11)
    assert pts.number_of_points == 1
    assert pts.coords[axis.vertical] == pint.Quantity(10.5, units["meter"])
    assert pts.coords[axis.latitude] == pint.Quantity(
        42.2, units["degrees_north"]
    )
    assert pts.coords[axis.longitude] == pint.Quantity(
        -12.2, units["degrees_east"]
    )


def test_nearest_vertical_points(points1):
    pts = points1.nearest_vertical(10.7)
    assert pts.number_of_points == 1
    assert pts.coords[axis.vertical] == pint.Quantity(10.5, units["meter"])
    assert pts.coords[axis.latitude] == pint.Quantity(
        42.2, units["degrees_north"]
    )
    assert pts.coords[axis.longitude] == pint.Quantity(
        -12.2, units["degrees_east"]
    )

    pts = points1.nearest_vertical(-100.0)
    assert pts.number_of_points == 1
    assert pts.coords[axis.vertical] == pint.Quantity(10.5, units["meter"])
    assert pts.coords[axis.latitude] == pint.Quantity(
        42.2, units["degrees_north"]
    )
    assert pts.coords[axis.longitude] == pint.Quantity(
        -12.2, units["degrees_east"]
    )


def test_time_range_points(points1):
    pts = points1.time_range("2001-01-01T08", "2001-01-02T04")
    assert pts.number_of_points == 1
    assert pts.coords[axis.vertical] == pint.Quantity(11.2, units["meter"])
    assert pts.coords[axis.latitude] == pint.Quantity(
        56.2, units["degrees_north"]
    )
    assert pts.coords[axis.longitude] == pint.Quantity(
        10.2, units["degrees_east"]
    )

    pts = points1.time_range("2000-01-01", "2001-01-01T04")
    assert pts.number_of_points == 1
    assert pts.coords[axis.vertical] == pint.Quantity(10.5, units["meter"])
    assert pts.coords[axis.latitude] == pint.Quantity(
        42.2, units["degrees_north"]
    )
    assert pts.coords[axis.longitude] == pint.Quantity(
        -12.2, units["degrees_east"]
    )


def test_latest_points(points1):
    pts = points1.latest()
    assert pts.number_of_points == 1
    assert pts.coords[axis.vertical] == pint.Quantity(11.2, units["meter"])
    assert pts.coords[axis.latitude] == pint.Quantity(
        56.2, units["degrees_north"]
    )
    assert pts.coords[axis.longitude] == pint.Quantity(
        10.2, units["degrees_east"]
    )


def test_to_netcdf_points(points1):
    path = "data/test_points.nc"
    points1.to_netcdf(path)
    dset = xr.load_dataset(path, decode_coords="all")
    pts_dset = points1._dset.rename(**{
        axis_: coord.attrs.get("standard_name", str(axis_))
        for axis_, coord in points1._dset.coords.items()
    })
    assert pts_dset.equals(dset)


#
# Profiles Tests
#


def test_create_profiles(profiles):
    assert profiles.dim_axes == (
        axis.time,
        axis.vertical,
        axis.latitude,
        axis.longitude,
    )
    assert profiles.number_of_profiles == 2
    assert profiles.number_of_levels == 4

    assert np.all(
        profiles.coords[axis.latitude]
        == pint.Quantity([42.2, 56.2], units["degrees_north"])
    )
    assert np.all(
        profiles.coords[axis.longitude]
        == pint.Quantity([-12.2, 10.2], units["degrees_east"])
    )
    assert np.allclose(
        profiles.coords[axis.vertical].magnitude,
        np.asarray([[10.5, 11.2, 12.3, np.nan], [10.7, 11.5, 12.5, 13.5]]),
        equal_nan=True,
    )
    assert profiles.coords[axis.vertical].units == units["m"]


def test_bounding_box_profiles(profiles, bbox_all, bbox_partial, bbox_noone):
    prof = profiles.bounding_box(**bbox_all)
    assert prof.number_of_profiles == 2
    assert prof.number_of_levels == 3

    prof = profiles.bounding_box(**bbox_partial)
    assert prof.number_of_profiles == 2
    assert prof.number_of_levels == 1


#    prof = profiles.bounding_box(**bbox_noone)
#    assert prof.number_of_levels == 0


def test_nearest_horizontal_profiles(profiles):
    prof = profiles.nearest_horizontal(latitude=41, longitude=-11)
    assert profiles.number_of_profiles == 2
    assert profiles.number_of_levels == 4


def test_nearest_vertical_points(profiles):
    prof = profiles.nearest_vertical(10.7)
    assert prof.number_of_profiles == 2
    assert prof.number_of_level == 1
    assert np.allclose(
        prof.coords[axis.vertical].to_numpy(),
        np.array([[np.nan], [10.7]]),
        equal_nan=True,
    )
    assert np.all(
        prof.coords[axis.time].to_numpy()
        == np.array(["2001-01-01", "2001-01-02"], dtype=np.datetime64)
    )
    assert np.allclose(
        prof.coords[axis.latitude].to_numpy(), np.array([42.2, 56.2])
    )
    assert np.allclose(
        prof.coords[axis.longitude].to_numpy(), np.array([-12.2, 10.2])
    )

    prof = profiles.nearest_vertical(-100.0)
    assert prof.number_of_profiles == 2
    assert prof.number_of_level == 1
    assert np.allclose(
        prof.coords[axis.vertical].to_numpy(),
        np.array([[np.nan], [10.7]]),
        equal_nan=True,
    )
    assert np.all(
        prof.coords[axis.time].to_numpy()
        == np.array(["2001-01-01", "2001-01-02"], dtype=np.datetime64)
    )
    assert np.allclose(
        prof.coords[axis.latitude].to_numpy(), np.array([42.2, 56.2])
    )
    assert np.allclose(
        prof.coords[axis.longitude].to_numpy(), np.array([-12.2, 10.2])
    )


def test_time_range_profiles(profiles):
    prof = profiles.time_range("2001-01-01T22", "2001-01-02")
    assert prof.number_of_profiles == 1
    assert prof.number_of_levels == 4
    assert np.allclose(
        prof.coords[axis.vertical].magnitude,
        np.array([[10.7, 11.5, 12.5, 13.5]]),
        equal_nan=True,
    )
    assert np.all(
        prof.coords[axis.time].magnitude
        == np.array(["2001-01-02"], dtype=np.datetime64)
    )
    assert np.allclose(prof.coords[axis.latitude].magnitude, np.array([56.2]))
    assert np.allclose(prof.coords[axis.longitude].magnitude, np.array([10.2]))


def test_latest_profiles(profiles):
    prof = profiles.latest()
    assert prof.number_of_profiles == 1
    assert prof.number_of_levels == 4
    assert np.allclose(
        prof.coords[axis.vertical].magnitude,
        np.array([[10.7, 11.5, 12.5, 13.5]]),
        equal_nan=True,
    )
    assert np.all(
        prof.coords[axis.time].magnitude
        == np.array(["2001-01-02"], dtype=np.datetime64)
    )
    assert np.allclose(prof.coords[axis.latitude].magnitude, np.array([56.2]))
    assert np.allclose(prof.coords[axis.longitude].magnitude, np.array([10.2]))


def test_to_netcdf_profiles(profiles):
    # FIXME: This giver a permission denied error.
    # path = 'data/test_profiles.nc'
    # profiles.to_netcdf(path)
    # dset = xr.load_dataset(path, decode_coords='all')
    # prof_dset = profiles._dset.rename(
    #     **{
    #         axis_: coord.attrs.get('standard_name', str(axis_))
    #         for axis_, coord in profiles._dset.coords.items()
    #     }
    # )
    # assert prof_dset.equals(dset)
    pass


#
# Grid Tests
#


def test_create_grid(grid):
    assert grid.dim_axes == (
        axis.time,
        axis.vertical,
        axis.latitude,
        axis.longitude,
    )

    # assert np.all(grid.coords[axis.latitude] == pint.Quantity([42.2, 56.2], units['degrees_north']))
    # assert np.all(grid.coords[axis.longitude] == pint.Quantity([-12.2, 10.2], units['degrees_east']))
    # assert np.all(grid.coords[axis.vertical] == pint.Quantity([10.5, 11.2, 12.3], units['meter']))


def test_bounding_box_grids(grid, bbox_all, bbox_partial, bbox_noone):
    grid_bb = grid.bounding_box(**bbox_all)
    # assert np.all(grid_bb.coords[axis.latitude] == pint.Quantity([42.2, 56.2], units['degrees_north']))
    # assert np.all(grid_bb.coords[axis.longitude] == pint.Quantity([-12.2, 10.2], units['degrees_east']))
    # assert np.all(grid_bb.coords[axis.vertical] == pint.Quantity([10.5, 11.2, 12.3], units['meter']))

    grid_bb = grid.bounding_box(**bbox_partial)


#    grid_bb = grid.bounding_box(**bbox_noone)


def test_nearest_horizontal_grid(grid):
    grid_ = grid.nearest_horizontal(latitude=41, longitude=-11)


def test_nearest_vertical_grid(grid):
    grid_ = grid.nearest_vertical(10.7)

    grid_ = grid.nearest_vertical(-100.0)


def test_time_range_grid(grid):
    pass


def test_latest_grid(grid):
    pass


def test_to_netcdf_grid(grid):
    grid.to_netcdf("test_grid.nc")
