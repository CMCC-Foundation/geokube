import pytest

import geokube.core.axis as axis
from geokube.core.coord_system import CoordinateSystem
from geokube.core.crs import Geodetic
from geokube.core.domain import Points, Profiles, Grid
from geokube.core.field import PointsField, ProfilesField, GridField
from geokube.core.units import units

import numpy as np

#
# bbox Fixtures
#


@pytest.fixture
def bbox_all():
    return {
        "south": 40,
        "north": 57,
        "east": 11,
        "west": -13,
        "top": 13,
        "bottom": 10,
    }


@pytest.fixture
def bbox_partial():
    return {
        "south": 40,
        "north": 57,
        "east": 11,
        "west": -13,
        "top": 11,
        "bottom": 10,
    }


@pytest.fixture
def bbox_noone():
    return {
        "south": 40,
        "north": 57,
        "east": 11,
        "west": -13,
        "top": 40,
        "bottom": 30,
    }


#
# Coordinate System Fixtures
#


@pytest.fixture
def geodetic_cs():
    return CoordinateSystem(
        horizontal=Geodetic(), elevation=axis.vertical, time=axis.time
    )


#
# Domain Fixtures
#


@pytest.fixture
def points1(geodetic_cs):
    pts = Points(
        coords=[
            ("2001-01-01", 10.5, 42.2, -12.2),
            ("2001-01-02", 11.2, 56.2, 10.2),
        ],
        coord_system=geodetic_cs,
    )
    return pts


@pytest.fixture
def points2(geodetic_cs):
    pts = Points(
        coords={
            axis.latitude: [42.2, 56.2],
            axis.longitude: [-12.2, 10.2],
            axis.vertical: [10.5, 11.2],
            axis.time: ["2001-01-01", "2001-01-02"],
        },
        coord_system=geodetic_cs,
    )
    return pts


@pytest.fixture
def profiles(geodetic_cs):
    prof = Profiles(
        coords={
            axis.latitude: [42.2, 56.2],
            axis.longitude: [-12.2, 10.2],
            axis.vertical: [[10.5, 11.2, 12.3], [10.7, 11.5, 12.5, 13.5]],
            axis.time: ["2001-01-01", "2001-01-02"],
        },
        coord_system=geodetic_cs,
    )
    return prof


@pytest.fixture
def grid(geodetic_cs):
    grid = Grid(
        coords={
            axis.latitude: [42.2, 56.2],
            axis.longitude: [-12.2, 10.2],
            axis.vertical: [10.5, 11.2, 12.3],
            axis.time: ["2001-01-01", "2001-01-02"],
        },
        coord_system=geodetic_cs,
    )
    return grid


#
# Field Fixtures
#


@pytest.fixture
def f_points_no_ancillary(points1):
    return PointsField(
        data=[22.5, 27.5] * units("degree_C"),
        domain=points1,
        name="field_points",
        properties={"test_property": "this is a field test property"},
    )


@pytest.fixture
def f_points(points1):
    return PointsField(
        data=[22.5, 27.5] * units("degree_C"),
        domain=points1,
        name="field_points",
        properties={"test_property": "this is a field test property"},
        ancillary={"anc_1": [0, 1] * units("")},
    )


@pytest.fixture
def f_profiles_no_ancillary(profiles):
    return ProfilesField(
        data=[[22.5, 27.5, 28.2, np.nan], [22.3, 26.2, 27.9, 29.1]]
        * units["degree_C"],
        domain=profiles,
        name="field_profiles",
        properties={"test_property": "this is a field test property"},
    )


@pytest.fixture
def f_profiles(profiles):
    return ProfilesField(
        data=[[22.5, 27.5, 28.2, np.nan], [22.3, 26.2, 27.9, 29.1]]
        * units["degree_C"],
        domain=profiles,
        name="field_profiles",
        properties={"test_property": "this is a field test property"},
        ancillary={"anc_profiles": [[0, 1, 2], [1, 0, 0, 2]] * units[""]},
    )


@pytest.fixture
def f_grid_no_ancillary(grid):
    a = np.arange(24).reshape(2, 3, 2, 2)
    return GridField(
        data=a * units("degree_C"),
        domain=grid,
        name="field_grid",
        properties={"test_property": "this is a field test property"},
    )


@pytest.fixture
def f_grid(grid):
    a = np.arange(24).reshape(2, 3, 2, 2)
    return GridField(
        data=a * units("degree_C"),
        domain=grid,
        name="field_grid",
        properties={"test_property": "this is a field test property"},
        ancillary={"anc_grid_1": a * units("")},
    )
