import pytest

import geokube.core.axis as axis 
from geokube.core.units import units
from geokube.core.axis import custom

import numpy as np

def test_predefined_axis_objects():
    assert axis.latitude
    assert axis.longitude
    assert axis.x
    assert axis.y
    assert axis.time

    assert axis.latitude.units == units['degrees_north']
    assert axis.latitude.units == axis.Latitude._DEFAULT_UNITS_


def test_userdefined_axes():
    uaxis = custom('test_axis')
    assert uaxis.units == units['']
    assert uaxis.dtype == np.dtype('str')