from numbers import Number

import dask.array as da
import numpy as np
import dask.array as da
import pytest

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, AxisType
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.unit import Unit
from geokube.core.variable import Variable
from tests import compare_dicts
from tests.fixtures import *


def test_process_bounds_fails():
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

    with pytest.raises(ex.HCubeValueError, match=r"Bounds should *"):
        _ = Coordinate._process_bounds(
            bounds=da.random.random((400, 2)),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )


def test_process_bounds_proper_attrs_setting():
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


def test_process_bounds_using_dict():
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


def test_init_fails():
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


def test_init_from_dask():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims=("latitude"))
    assert c.dim_names == ("latitude",)
    assert c.dim_ncvars == ("latitude",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LATITUDE
    assert c.is_independent


@pytest.mark.skip(
    "Invalidate as in the current version, if `dims` is None, it is created based on provided `axis`"
)
def test_init_from_dask_fail():
    D = da.random.random((100,))

    with pytest.raises(ex.HCubeValueError, match=r"Provided data have *"):
        _ = Coordinate(
            data=D,
            axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
            dims=None,
        )


def test_init_from_dask_proper_attrs_setting():
    D = da.random.random((100,))
    c = Coordinate(
        data=D,
        axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
        dims="lon",
    )
    assert c.name == "lon"
    assert c.ncvar == "new_lon_name"
    assert c.dim_names == ("lon",)
    assert c.dim_ncvars == ("lon",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LONGITUDE
    assert c.is_independent


def test_init_from_dask_fail_on_bounds_shape():
    D = da.random.random((100,))
    with pytest.raises(ex.HCubeValueError, match=r"Bounds should*"):
        _ = Coordinate(
            data=D,
            axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
            bounds=np.ones((104, 2)),
        )


def test_init_from_numpy_proper_attrs_setting():
    D = np.random.random((100,))
    c = Coordinate(
        data=D,
        axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
        dims="lon",
        encoding={"name": "my_lon_name"},
    )
    assert c.name == "lon"
    # if encoding provieded for Axis and Cooridnate, they are merged. Keys in Axis encoding will be overwritten
    assert c.ncvar == "my_lon_name"
    assert c.dim_names == ("lon",)
    assert c.dim_ncvars == ("lon",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LONGITUDE
    assert c.is_independent


def test_init_fails_on_scalar_data_if_dims_passed():
    with pytest.raises(ex.HCubeValueError, match=r"Provided data have *"):
        _ = Coordinate(data=10, axis="longitude", dims="longitude")


def test_init_with_scalar_data():
    c = Coordinate(data=10, axis="longitude")
    assert c.type is CoordinateType.SCALAR
    assert c.axis_type is AxisType.LONGITUDE
    assert np.all(c.values == 10)
    assert c.is_independent  # Scalar treated as independent


def test_init_fails_on_missing_dim():
    D = da.random.random((100, 50))
    with pytest.raises(ex.HCubeValueError, match=r"Provided data have *"):
        _ = Coordinate(data=D, axis="longitude", dims="longitude")


def test_init_proper_multidim_coord():
    D = da.random.random((100, 50))
    c = Coordinate(data=D, axis="longitude", dims=["x", "y"])
    assert c.type is CoordinateType.DEPENDENT
    assert c.axis_type is AxisType.LONGITUDE
    assert c.dim_names == ("x", "y")
    assert c.dim_ncvars == ("x", "y")
    assert c.is_dependent


def test_from_xarray__regular_latlon(era5_netcdf):
    c = Coordinate.from_xarray(era5_netcdf, "time")
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.TIME
    assert c.dim_names == ("time",)
    assert c.units == Unit(
        era5_netcdf["time"].encoding["units"], era5_netcdf["time"].encoding["calendar"]
    )
    assert c.bounds is None
    assert not c.has_bounds


def test_from_xarray_rotated_pole(era5_rotated_netcdf):
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


def test_from_xarray_rotated_pole_with_mapping(era5_rotated_netcdf):
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


def test_to_xarray_rotated_pole_without_encoding(era5_rotated_netcdf):
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


def test_to_xarray_rotated_pole_with_encoding(era5_rotated_netcdf):
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


def test_to_xarray_rotated_pole_with_encoding_2(era5_rotated_netcdf):
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


def test_to_xarray_rotated_pole_without_encoding_2(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "lat")
    res = c.to_xarray(encoding=True)
    assert res.name == "lat"
    assert "rlat" in res.dims
    assert "rlon" in res.dims
    assert np.all(era5_rotated_netcdf.lat.values == res.lat.values)
    assert res.attrs == era5_rotated_netcdf.lat.attrs
    assert set(res.encoding) - {"name"} == set(
        era5_rotated_netcdf.lat.encoding.keys()
    ) - {"bounds"}


def test_toxarray_keeping_encoding_encoding_false_no_dims_passed():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("lat", is_dim=True))
    coord = c.to_xarray(encoding=False)
    assert "lat" in coord.coords
    assert "latitude" not in coord.coords
    assert coord.name == "lat"
    assert coord.dims == ("lat",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "lat"

    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True))
    coord = c.to_xarray(encoding=False)
    assert "latitude" in coord.coords
    assert coord.name == "latitude"
    assert "lat" not in coord.coords
    assert coord.dims == ("latitude",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "latitude"


def test_toxarray_keeping_encoding_encoding_false():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("lat", is_dim=True), dims=("lat"))
    coord = c.to_xarray(encoding=False)
    assert "lat" in coord.coords
    assert coord.name == "lat"
    assert coord.dims == ("lat",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "lat"

    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims=("latitude"))
    coord = c.to_xarray(encoding=False)
    assert coord.dims == ("latitude",)
    assert coord.name == "latitude"
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "latitude"


def test_init_fails_if_is_dim_and_axis_name_differ_from_dims():
    D = da.random.random((100,))
    with pytest.raises(
        ex.HCubeValueError,
        match=r"If the Coordinate is a dimension, it has to depend only on itself, but provided `dims` are*",
    ):
        _ = Coordinate(data=D, axis=Axis("lat", is_dim=True), dims=("x", "y"))

    with pytest.raises(
        ex.HCubeValueError,
        match=r"`dims` parameter for dimension coordinate should have the same name as axis name*",
    ):
        _ = Coordinate(data=D, axis=Axis("lat", is_dim=True), dims=("latitude"))

    with pytest.raises(
        ex.HCubeValueError,
        match=r"`dims` parameter for dimension coordinate should have the same name as axis name*",
    ):
        _ = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims=("lat"))


def test_toxarray_keeping_encoding_encoding_true():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("lat", is_dim=True))
    coord = c.to_xarray(encoding=True)
    assert coord.name == "lat"
    assert coord.dims == ("lat",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "lat"

    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims=("latitude"))
    coord = c.to_xarray(encoding=True)
    assert coord.name == "latitude"
    assert coord.dims == ("latitude",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "latitude"


def test_coord_data_always_numpy_array(era5_rotated_netcdf, era5_netcdf):
    for c in era5_rotated_netcdf.coords.keys():
        coord = Coordinate.from_xarray(era5_rotated_netcdf, c)
        assert isinstance(coord._data, np.ndarray)

    for c in era5_netcdf.coords.keys():
        coord = Coordinate.from_xarray(era5_netcdf, c)
        assert isinstance(coord._data, np.ndarray)

    D = da.random.random((100,))
    coord = Coordinate(
        data=D,
        axis=Axis("lon2", is_dim=True, encoding={"name": "new_lon_name"}),
        dims="lon2",
        encoding={"name": "my_lon_name"},
    )

    assert isinstance(coord._data, np.ndarray)

    D = np.random.random((100,))
    coord = Coordinate(
        data=D,
        axis=Axis("lon", is_dim=True, encoding={"name": "new_lon_name"}),
        dims="lon",
        encoding={"name": "my_lon_name"},
    )

    assert isinstance(coord._data, np.ndarray)


def test_to_xarray_with_bounds(era5_rotated_netcdf, nemo_ocean_16):
    coord = Coordinate.from_xarray(era5_rotated_netcdf, "time")
    da, bounds = coord._get_xarray_and_bounds(encoding=False)
    assert "time_bnds" in bounds
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bnds"

    coord = Coordinate.from_xarray(nemo_ocean_16, "time_counter")
    da, bounds = coord._get_xarray_and_bounds(encoding=False)
    assert "time_counter_bounds" in bounds
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_counter_bounds"

    da, bounds = coord._get_xarray_and_bounds(encoding=True)
    assert "time_counter_bounds" in bounds
    assert "bounds" in da["time_counter"].encoding
    assert da["time_counter"].encoding["bounds"] == "time_counter_bounds"

    coord = Coordinate.from_xarray(nemo_ocean_16, "nav_lat")
    da, bounds = coord._get_xarray_and_bounds(encoding=False)
    assert "bounds_lat" in bounds
    assert "bounds" in da["latitude"].encoding
    assert da["latitude"].encoding["bounds"] == "bounds_lat"

    da, bounds = coord._get_xarray_and_bounds(encoding=True)
    assert "bounds_lat" in bounds
    assert "bounds" in da["nav_lat"].encoding
    assert da["nav_lat"].encoding["bounds"] == "bounds_lat"

    coord = Coordinate.from_xarray(nemo_ocean_16, "nav_lon")
    da, bounds = coord._get_xarray_and_bounds(encoding=False)
    assert "bounds_lon" in bounds
    assert "bounds" in da["longitude"].encoding
    assert da["longitude"].encoding["bounds"] == "bounds_lon"

    da, bounds = coord._get_xarray_and_bounds(encoding=True)
    assert "bounds_lon" in bounds
    assert "bounds" in da["nav_lon"].encoding
    assert da["nav_lon"].encoding["bounds"] == "bounds_lon"
