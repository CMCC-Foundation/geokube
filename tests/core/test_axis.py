import pytest

from geokube.core.axis import Axis, AxisType
from geokube.core.unit import Unit
from tests.fixtures import *


def test_init_proper_attributes_set():
    assert AxisType.LATITUDE.default_unit == Unit("degrees_north")
    assert AxisType.TIME.axis_type_name == "time"
    assert AxisType("aaa") is AxisType.GENERIC
    assert AxisType.GENERIC.default_unit == Unit("unknown")

    a1 = Axis(name="LAT", axistype="latitude")

    assert a1.name == "LAT"
    assert a1.type is AxisType.LATITUDE
    assert a1.encoding is None


def test_parsing():
    assert AxisType.parse("latitude") is AxisType.LATITUDE
    assert AxisType.parse("lat") is AxisType.LATITUDE
    assert AxisType.parse("rlat") is AxisType.Y
    assert AxisType.parse("x").default_unit == Unit("m")
    assert AxisType.parse("depth") is AxisType.VERTICAL
    assert AxisType.parse("time").default_unit == Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )


def test_init_fails():
    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `str`,"
            r" `geokube.AxisType`, but provided *"
        ),
    ):
        _ = Axis("lon", axistype=10)

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `str`,"
            r" `geokube.AxisType`, but provided *"
        ),
    ):
        _ = Axis("lon", axistype={"lat"})

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `str`,"
            r" `geokube.AxisType`, but provided *"
        ),
    ):
        _ = Axis("lon", axistype=["lon"])


def test_init_from_other_axis():
    a1 = Axis(name="LAT", axistype="latitude")
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


def test_axis_hash():
    a1 = Axis(AxisType.LONGITUDE)
    a2 = Axis(AxisType.LONGITUDE)
    assert a1 == a2
    assert hash(a1) == hash(a2)

    a1 = Axis("longitude")
    a2 = Axis(name="longitude", axistype=AxisType.LONGITUDE)
    assert a1 == a2
    assert hash(a1) == hash(a2)

    a1 = Axis("longitude", is_dim=True)
    a2 = Axis(AxisType.LONGITUDE, is_dim=False)
    assert a1 != a2
    assert hash(a1) != hash(a2)

    a1 = Axis("lat", AxisType.LONGITUDE)
    a2 = Axis(AxisType.LONGITUDE)
    assert a1 != a2
    assert hash(a1) != hash(a2)

    d = {Axis("lat"): [1, 2, 3], Axis("depth"): [-1, -2, -3]}
    assert Axis("lat") in d
    assert Axis("depth") in d
    assert d[Axis("lat")] == [1, 2, 3]
    assert d[Axis("depth")] == [-1, -2, -3]


def test_vertical_axis_pattern():
    assert Axis("dept").type is AxisType.VERTICAL
    assert Axis("depth").type is AxisType.VERTICAL
