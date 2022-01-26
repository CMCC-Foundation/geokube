import pytest
from cf_units import Unit

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, Axis, AxisType
from tests.fixtures import *


def test_axis_type():
    assert AxisType.LATITUDE.default_unit == Unit("degrees_north")
    assert AxisType.TIME.axis_type_name == "time"
    assert AxisType("aaa") is AxisType.GENERIC
    assert AxisType.parse("latitude") is AxisType.LATITUDE
    assert AxisType.parse("lat") is AxisType.LATITUDE
    assert AxisType.parse("rlat") is AxisType.Y
    assert AxisType.parse("x").default_unit == Unit("m")
    assert AxisType.parse("depth") is AxisType.VERTICAL
    assert AxisType.parse("time").default_unit == Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )
    assert AxisType.GENERIC.default_unit == Unit("unknown")


def test_axis_2():
    a1 = Axis(name="LAT", axistype="latitude")

    assert a1.name == "LAT"
    assert a1.type is AxisType.LATITUDE
    assert a1.encoding is None

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected type: str or geokube.AxisType, provided type *",
    ):
        _ = Axis("lon", axistype=10)

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected type: str or geokube.AxisType, provided type *",
    ):
        _ = Axis("lon", axistype={"lat"})

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected type: str or geokube.AxisType, provided type *",
    ):
        _ = Axis("lon", axistype=["lon"])

    a3 = Axis(a1)
    assert id(a3) != id(a1)
    assert a3.name == a1.name
    assert a3.type is a1.type
    assert a3.encoding == a1.encoding
    assert a3.default_unit == Unit("degrees_north")

    a4 = Axis("lon", encoding={"name": "ncvar_my_name"})
    assert a4.name == "lon"
    assert a4.type is AxisType.LONGITUDE
    assert a4.encoding == {"name": "ncvar_my_name"}
    assert a4.default_unit == Unit("degrees_east")
    assert a4.ncvar == "ncvar_my_name"
    assert not a4.is_dim

    a5 = Axis(a4)
    assert a5.name == a4.name
    assert a5.type is a4.type
    assert a5.encoding == a4.encoding
    assert a5.default_unit == Unit("degrees_east")
    assert a5.ncvar == "ncvar_my_name"
    assert not a5.is_dim

    a6 = Axis("time", is_dim=True)
    assert a6.name == "time"
    assert a6.type is AxisType.TIME
    assert a6.is_dim
