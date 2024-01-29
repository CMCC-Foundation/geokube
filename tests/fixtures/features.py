import pytest

import geokube.core.axis as axis
from geokube.core.coord_system import CoordinateSystem
from geokube.core.crs import Geodetic
from geokube.core.domain import Points, Profiles, Grid
from geokube.core.field import PointsField, ProfilesField, GridField
from geokube.core.units import units

#
# bbox Fixtures
# 

@pytest.fixture
def bbox_all():
    return {'south':40, 'north':57, 'east':11, 'west':-13, 'top':13, 'bottom':10}

@pytest.fixture
def bbox_partial():
    return {'south':40, 'north':57, 'east':11, 'west':-13, 'top':11, 'bottom':10}

@pytest.fixture
def bbox_noone():
    return {'south':40, 'north':57, 'east':11, 'west':-13, 'top':40, 'bottom':30}

#
# Coordinate System Fixtures
# 

@pytest.fixture
def geodetic_cs():
    return CoordinateSystem(
        horizontal = Geodetic(),
        elevation = axis.vertical,
#       time = axis.time
    )

#
# Domain Fixtures
# 

@pytest.fixture
def points1(geodetic_cs):
    pts = Points(
        coords = [(10.5, 42.2, -12.2), (11.2, 56.2, 10.2)],
        coord_system = geodetic_cs
    )
    return pts
    
@pytest.fixture
def points2(geodetic_cs):
    pts = Points(
        coords = {
            axis.latitude: [42.2, 56.2],
            axis.longitude: [-12.2, 10.2],
            axis.vertical: [10.5, 11.2]
        },
        coord_system = geodetic_cs
    )
    return pts

@pytest.fixture
def profiles(geodetic_cs):
    prof = Profiles(
        coords = {
            axis.latitude: [42.2, 56.2],
            axis.longitude: [-12.2, 10.2],
            axis.vertical: [[10.5, 11.2, 12.3], [10.7, 11.5, 12.5, 13.5]]
        },
        coord_system = geodetic_cs
    )
    return prof

@pytest.fixture
def grid(geodetic_cs):
    grid = Grid(
        coords = {
            axis.latitude: [42.2, 56.2],
            axis.longitude: [-12.2, 10.2],
            axis.vertical: [10.5, 11.2, 12.3]
        },
        coord_system = geodetic_cs
    )
    return grid


#
# Field Fixtures
# 

@pytest.fixture
def f_points_no_ancillary(points1):
    return PointsField(
        data = [22.5, 27.5] * units['degree_C'],
        domain = points1,
        name = 'field_1',
        properties = {'test_property': 'this is a field test property'},
    )

@pytest.fixture
def f_points(points1):
    return PointsField(
        data = [22.5, 27.5] * units['degree_C'],
        domain = points1,
        name = 'field_1',
        properties = {'test_property': 'this is a field test property'},
        ancillary = {
            'anc_1': [0, 1] * units['']
        },
    )
