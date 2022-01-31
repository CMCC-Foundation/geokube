from numbers import Number
import numpy as np
import dask.array as da
import pytest

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.core.axis import Axis, Axis, AxisType
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.variable import Variable
from tests.fixtures import *
from tests import compare_dicts


def test_process_bounds_1():
    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected argument is one of the following types `dict`, `numpy.ndarray`, or `geokube.Variable`, but provided*",
    ):
        Coordinate._process_bounds(
            [1, 2, 3, 4],
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected argument is one of the following types `dict`, `numpy.ndarray`, or `geokube.Variable`, but provided*",
    ):
        Coordinate._process_bounds(
            "bounds",
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected argument is one of the following types `dict`, `numpy.ndarray`, or `geokube.Variable`, but provided*",
    ):
        Coordinate._process_bounds(
            xr.DataArray([1, 2, 3, 4]),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )


def test_process_bounds_2():
    with pytest.raises(ex.HCubeValueError, match=r"Bounds should *"):
        _ = Coordinate._process_bounds(
            Variable(data=np.random.rand(100, 5), dims=["time", "bounds"]),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(ex.HCubeValueError, match=r"Bounds should *"):
        _ = Coordinate._process_bounds(
            Variable(data=np.random.rand(100, 5), dims=["time", "bounds"], units="m"),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    D = np.random.rand(100, 2)
    r = Coordinate._process_bounds(
        Variable(data=D, dims=["time", "bounds"]),
        name="name",
        units="units",
        axis=Axis("time"),
        variable_shape=(100, 1),
    )
    assert isinstance(r, dict)
    assert "name_bounds" in r
    assert np.all(r["name_bounds"] == D)

    r = Coordinate._process_bounds(
        bounds=D, name="name", units="units", axis=Axis("time"), variable_shape=(100, 1)
    )
    assert isinstance(r, dict)
    assert "name_bounds" in r
    assert isinstance(r["name_bounds"], Variable)
    assert r["name_bounds"].units == Unit("units")
    assert np.all(r["name_bounds"] == D)

    with pytest.raises(ex.HCubeValueError, match=r"Bounds should *"):
        _ = Coordinate._process_bounds(
            bounds=da.random.random((400, 2)),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    D = da.random.random((400, 2))
    r = Coordinate._process_bounds(
        bounds=D,
        name="name2",
        units="units",
        axis=Axis("time"),
        variable_shape=(400, 1),
    )
    assert isinstance(r, dict)
    assert "name2_bounds" in r
    assert isinstance(r["name2_bounds"], Variable)
    assert r["name2_bounds"].units == Unit("units")
    assert np.all(r["name2_bounds"] == D)


def test_process_bounds_3():
    d = {
        "q": np.ones((100, 2)),
        "w_bounds": Variable(
            data=np.full((100, 2), fill_value=10), dims=("lat", "bounds")
        ),
    }

    # with pytest.raises(ex.HCubeValueError, match=r"Bounds should*"):
    #     _ = Coordinate._process_bounds(
    #         d, name="name2", units="m", axis=Axis("lat"), variable_shape=(400, 1)
    #     )

    r = Coordinate._process_bounds(
        d, name="name2", units="m", axis=Axis("lon"), variable_shape=(100, 1)
    )
    assert "q" in r
    assert "w_bounds" in r
    assert r["q"].units == Unit("m")
    assert set(r["q"].dim_names) == {"lon", "bounds"}
    assert r["w_bounds"].units == Unit(
        None
    )  # no unit provided in `w` Variable definition
    assert set(r["w_bounds"].dim_names) == {
        "lat",
        "bounds",
    }  # `lat` defined as `w` Variable dim


def test_init_1():
    with pytest.raises(ex.HCubeValueError, match=r"`data` cannot be `None`"):
        _ = Coordinate(data=None, axis="lat")

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected argument is one of the following types `geokube.Axis` or `str`, but provided *",
    ):
        _ = Coordinate(data=np.ones(100), axis=["lat"])

    with pytest.raises(
        ex.HCubeTypeError,
        match=r"Expected argument is one of the following types `geokube.Axis` or `str`, but provided *",
    ):
        _ = Coordinate(data=np.ones(100), axis=15670)

    with pytest.raises(
        ex.HCubeValueError,
        match=r"If coordinate is not a dimension, you need to supply `dims` argument!",
    ):
        _ = Coordinate(data=np.ones(100), axis=Axis("lat", is_dim=False))


def test_init_2():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims=("latitude"))
    assert c.dim_names == ("latitude",)
    assert c.dim_ncvars == ("latitude",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LATITUDE
    assert c.is_independent


def test_init_3():
    D = da.random.random((100,))

    with pytest.raises(ex.HCubeValueError, match=r"Provided data have *"):
        _ = Coordinate(
            data=D,
            axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
            dims=None,
        )


def test_init_4():
    D = da.random.random((100,))
    c = Coordinate(
        data=D,
        axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
        dims="lon2",
    )
    assert c.name == "lon"
    assert c.ncvar == "new_lon_name"
    assert c.dim_names == ("lon2",)
    assert c.dim_ncvars == ("lon2",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LONGITUDE
    assert c.is_independent


def test_init_5():
    D = da.random.random((100,))
    with pytest.raises(ex.HCubeValueError, match=r"Bounds should*"):
        _ = Coordinate(
            data=D,
            axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
            dims="lon2",
            bounds=np.ones((104, 2)),
        )
    c = Coordinate(
        data=D,
        axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
        dims="lon2",
        encoding={"name": "my_lon_name"},
    )
    assert c.name == "lon"
    # if encoding provieded for Axis and Cooridnate, they are merged. Keys in Axis encoding will be overwritten
    assert c.ncvar == "my_lon_name"
    assert c.dim_names == ("lon2",)
    assert c.dim_ncvars == ("lon2",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LONGITUDE
    assert c.is_independent


def test_init_6():
    with pytest.raises(ex.HCubeValueError, match=r"Provided data have *"):
        _ = Coordinate(data=10, axis="longitude", dims="longitude")

    c = Coordinate(data=10, axis="longitude")
    assert c.type is CoordinateType.SCALAR
    assert c.axis_type is AxisType.LONGITUDE
    assert np.all(c.values == 10)
    assert c.is_independent  # Scalar treated as independent


def test_init_7():
    D = da.random.random((100, 50))
    with pytest.raises(ex.HCubeValueError, match=r"Provided data have *"):
        _ = Coordinate(data=D, axis="longitude", dims="longitude")

    c = Coordinate(data=D, axis="longitude", dims=["x", "y"])
    assert c.type is CoordinateType.DEPENDENT
    assert c.axis_type is AxisType.LONGITUDE
    assert c.dim_names == ("x", "y")
    assert c.dim_ncvars == ("x", "y")
    assert c.is_dependent


def test_from_xarray_1(era5_netcdf):
    c = Coordinate.from_xarray(era5_netcdf, "time")
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.TIME
    assert c.dim_names == ("time",)
    assert c.units == Unit(
        era5_netcdf["time"].encoding["units"], era5_netcdf["time"].encoding["calendar"]
    )
    assert c.bounds is None
    assert not c.has_bounds


def test_from_xarray_2(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "soil1")
    assert c.dim_names == ("depth",)
    assert c.dim_ncvars == ("soil1",)
    assert c.has_bounds
    assert c.bounds is not None
    assert c.name == "depth"
    assert c.ncvar == "soil1"
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.VERTICAL or c.axis_type is AxisType.Z
    assert c.dim_names == ("depth",)
    assert c.dim_ncvars == ("soil1",)
    assert c.units == Unit("m")
    assert c.bounds["soil1_bnds"].units == Unit("m")


def test_from_xarray_3(era5_rotated_netcdf):
    c = Coordinate.from_xarray(
        era5_rotated_netcdf, "soil1", mapping={"soil1": {"name": "new_soil"}}
    )
    assert c.has_bounds
    assert c.bounds is not None
    assert c.name == "new_soil"
    assert c.ncvar == "soil1"
    assert c.dim_names == ("new_soil",)
    assert c.dim_ncvars == ("soil1",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.VERTICAL or c.axis_type is AxisType.Z
    assert c.dim_ncvars == ("soil1",)
    assert c.units == Unit("m")
    assert c.bounds["soil1_bnds"].units == Unit("m")


def test_to_xarray_1(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "soil1")
    res = c.to_xarray(encoding=False)

    assert res.name == "depth"
    assert np.all(era5_rotated_netcdf.soil1.values == res.depth.values)
    assert res.attrs == era5_rotated_netcdf.soil1.attrs
    assert set(res.encoding) - {"name"} == set(
        era5_rotated_netcdf.soil1.encoding.keys()
    ) - {"bounds"}

    # with simple `to_xarray` bounds are not returned and shouldn't be recorded
    assert "bounds" not in res.encoding
    assert "bounds" not in res.attrs
    compare_dicts(
        res.encoding,
        era5_rotated_netcdf.soil1.encoding,
        exclude_d1="name",
        exclude_d2="bounds",
    )  # TODO: currently bounds are not included


def test_to_xarray_2(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "soil1")
    res = c.to_xarray(encoding=True)

    assert res.name == "soil1"
    assert np.all(era5_rotated_netcdf.soil1.values == res.soil1.values)
    assert res.attrs == era5_rotated_netcdf.soil1.attrs
    assert set(res.encoding) - {"name"} == set(
        era5_rotated_netcdf.soil1.encoding.keys()
    ) - {"bounds"}
    compare_dicts(
        res.encoding,
        era5_rotated_netcdf.soil1.encoding,
        exclude_d1="name",
        exclude_d2="bounds",
    )  # TODO: currently bounds are not included


def test_to_xarray_3(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "lat")
    assert c.type is CoordinateType.DEPENDENT
    res = c.to_xarray(encoding=False)
    assert res.name == "latitude"
    assert "grid_latitude" in res.dims
    assert "grid_longitude" in res.dims
    assert np.all(era5_rotated_netcdf.lat.values == res.latitude.values)
    assert res.attrs == era5_rotated_netcdf.lat.attrs
    assert set(res.encoding) - {"name"} == set(
        era5_rotated_netcdf.lat.encoding.keys()
    ) - {"bounds"}

    res = c.to_xarray(encoding=True)
    assert res.name == "lat"
    assert "rlat" in res.dims
    assert "rlon" in res.dims
    assert np.all(era5_rotated_netcdf.lat.values == res.lat.values)
    assert res.attrs == era5_rotated_netcdf.lat.attrs
    assert set(res.encoding) - {"name"} == set(
        era5_rotated_netcdf.lat.encoding.keys()
    ) - {"bounds"}
