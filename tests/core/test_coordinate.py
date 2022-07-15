from numbers import Number

import dask.array as da
import numpy as np
import dask.array as da
import pytest

from geokube.core.axis import Axis, AxisType
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.unit import Unit
from geokube.core.variable import Variable
from tests import compare_dicts
from tests.fixtures import *


def test_process_bounds_fails():
    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `dict`,"
            r" `numpy.ndarray`, or `geokube.Variable`, but provided*"
        ),
    ):
        Coordinate._process_bounds(
            [1, 2, 3, 4],
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `dict`,"
            r" `numpy.ndarray`, or `geokube.Variable`, but provided*"
        ),
    ):
        Coordinate._process_bounds(
            "bounds",
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `dict`,"
            r" `numpy.ndarray`, or `geokube.Variable`, but provided*"
        ),
    ):
        Coordinate._process_bounds(
            xr.DataArray([1, 2, 3, 4]),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(ValueError, match=r"Bounds should *"):
        _ = Coordinate._process_bounds(
            Variable(data=np.random.rand(100, 5), dims=["time", "bounds"]),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(ValueError, match=r"Bounds should *"):
        _ = Coordinate._process_bounds(
            Variable(
                data=np.random.rand(100, 5), dims=["time", "bounds"], units="m"
            ),
            name="name",
            units="units",
            axis=Axis("time"),
            variable_shape=(100, 1),
        )

    with pytest.raises(ValueError, match=r"Bounds should *"):
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
        bounds=D,
        name="name",
        units="units",
        axis=Axis("time"),
        variable_shape=(100, 1),
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

    # with pytest.raises(ValueError, match=r"Bounds should*"):
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
    with pytest.raises(ValueError, match=r"`data` cannot be `None`"):
        _ = Coordinate(data=None, axis="lat")

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `geokube.Axis`"
            r" or `str`, but provided *"
        ),
    ):
        _ = Coordinate(data=np.ones(100), axis=["lat"])

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `geokube.Axis`"
            r" or `str`, but provided *"
        ),
    ):
        _ = Coordinate(data=np.ones(100), axis=15670)

    with pytest.raises(
        ValueError,
        match=(
            r"If coordinate is not a dimension, you need to supply `dims`"
            r" argument!"
        ),
    ):
        _ = Coordinate(data=np.ones(100), axis=Axis("lat", is_dim=False))


def test_init_from_dask():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims="latitude")
    assert c.dim_names == ("latitude",)
    assert c.dim_ncvars == ("latitude",)
    assert c.type is CoordinateType.INDEPENDENT
    assert c.axis_type is AxisType.LATITUDE
    assert c.is_independent


@pytest.mark.skip(
    "Invalidate as in the current version, if `dims` is None, it is created"
    " based on provided `axis`"
)
def test_init_from_dask_fail():
    D = da.random.random((100,))

    with pytest.raises(ValueError, match=r"Provided data have *"):
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
    with pytest.raises(ValueError, match=r"Bounds should*"):
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


def test_init_with_scalar_data():
    c = Coordinate(data=10, axis="longitude")
    assert c.type is CoordinateType.SCALAR
    assert c.axis_type is AxisType.LONGITUDE
    assert np.all(c.values == 10)
    assert c.is_independent  # Scalar treated as independent


def test_init_fails_on_missing_dim():
    D = da.random.random((100, 50))
    with pytest.raises(ValueError, match=r"Provided data have *"):
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
        era5_netcdf["time"].encoding["units"],
        era5_netcdf["time"].encoding["calendar"],
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
    coord_dset = c.to_xarray(encoding=False)
    coord = coord_dset["depth"]

    assert coord.name == "depth"
    assert np.all(era5_rotated_netcdf.soil1.values == coord.depth.values)
    assert coord.attrs == era5_rotated_netcdf.soil1.attrs

    assert ("bounds" in coord.encoding) or ("bounds" in coord.attrs)
    compare_dicts(
        coord.encoding,
        era5_rotated_netcdf.soil1.encoding,
        exclude_d1=["name", "_FillValue"],
        exclude_d2="_FillValue",
    )
    assert coord.encoding["_FillValue"] is None


def test_to_xarray_rotated_pole_with_encoding(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "soil1")
    coord_dset = c.to_xarray(encoding=True)
    coord = coord_dset["soil1"]

    assert coord.name == "soil1"
    assert np.all(era5_rotated_netcdf.soil1.values == coord.soil1.values)
    assert coord.attrs == era5_rotated_netcdf.soil1.attrs
    compare_dicts(
        coord.encoding,
        era5_rotated_netcdf.soil1.encoding,
        exclude_d1=["name", "_FillValue"],
        exclude_d2="_FillValue",
    )
    assert coord.encoding["_FillValue"] is None


def test_to_xarray_rotated_pole_with_encoding_2(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "lat")
    assert c.type is CoordinateType.DEPENDENT
    coord_dset = c.to_xarray(encoding=False)
    coord = coord_dset["latitude"]
    assert coord.name == "latitude"
    assert "grid_latitude" in coord.dims
    assert "grid_longitude" in coord.dims
    assert np.all(era5_rotated_netcdf.lat.values == coord.latitude.values)
    assert coord.attrs == era5_rotated_netcdf.lat.attrs
    assert set(coord.encoding) - {"name", "_FillValue"} == set(
        era5_rotated_netcdf.lat.encoding.keys()
    ) - {"bounds"}


def test_to_xarray_rotated_pole_without_encoding_2(era5_rotated_netcdf):
    c = Coordinate.from_xarray(era5_rotated_netcdf, "lat")
    coord_dset = c.to_xarray(encoding=True)
    coord = coord_dset["lat"]
    assert coord.name == "lat"
    assert "rlat" in coord.dims
    assert "rlon" in coord.dims
    assert np.all(era5_rotated_netcdf.lat.values == coord.lat.values)
    assert coord.attrs == era5_rotated_netcdf.lat.attrs
    assert set(coord.encoding) - {"name", "_FillValue"} == set(
        era5_rotated_netcdf.lat.encoding.keys()
    ) - {"bounds"}


def test_toxarray_keeping_encoding_encoding_false_no_dims_passed():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("lat", is_dim=True))
    coord_dset = c.to_xarray(encoding=False)
    coord = coord_dset["lat"]
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
    coord_dset = c.to_xarray(encoding=False)
    coord = coord_dset["latitude"]
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
    c = Coordinate(data=D, axis=Axis("lat", is_dim=True), dims="lat")
    coords_dset = c.to_xarray(encoding=False)
    coord = coords_dset["lat"]
    assert "lat" in coord.coords
    assert coord.name == "lat"
    assert coord.dims == ("lat",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "lat"

    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims="latitude")
    coords_dset = c.to_xarray(encoding=False)
    coord = coords_dset["latitude"]
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
        ValueError,
        match=(
            r"If the Coordinate is a dimension, it has to depend only on"
            r" itself, but provided `dims` are*"
        ),
    ):
        _ = Coordinate(data=D, axis=Axis("lat", is_dim=True), dims=("x", "y"))

    with pytest.raises(
        ValueError,
        match=(
            r"`dims` parameter for dimension coordinate should have the same"
            r" name as axis name*"
        ),
    ):
        _ = Coordinate(data=D, axis=Axis("lat", is_dim=True), dims="latitude")

    with pytest.raises(
        ValueError,
        match=(
            r"`dims` parameter for dimension coordinate should have the same"
            r" name as axis name*"
        ),
    ):
        _ = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims="lat")


def test_toxarray_keeping_encoding_encoding_true():
    D = da.random.random((100,))
    c = Coordinate(data=D, axis=Axis("lat", is_dim=True))
    coord_dset = c.to_xarray(encoding=True)
    coord = coord_dset["lat"]
    assert coord.name == "lat"
    assert coord.dims == ("lat",)
    assert "standard_name" in coord.attrs
    assert coord.attrs["standard_name"] == "latitude"
    assert "units" in coord.attrs
    assert coord.attrs["units"] == "degrees_north"
    assert "name" in coord.encoding
    assert coord.encoding["name"] == "lat"

    c = Coordinate(data=D, axis=Axis("latitude", is_dim=True), dims="latitude")
    coord_dset = c.to_xarray(encoding=True)
    coord = coord_dset["latitude"]
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


def test_vertical_pattern_model_level_number(nemo_ocean_16):
    nemo_ocean_16.depthv.attrs["standard_name"] = "model_level_number"
    coord = Coordinate.from_xarray(nemo_ocean_16, "depthv")
    assert coord.axis_type is AxisType.VERTICAL


def test_to_xarray_with_bounds(era5_rotated_netcdf, nemo_ocean_16):
    coord = Coordinate.from_xarray(era5_rotated_netcdf, "time")
    coord_dset = coord.to_xarray(encoding=False)
    assert "time_bnds" in coord_dset
    assert "bounds" in coord_dset["time"].encoding
    assert coord_dset["time"].encoding["bounds"] == "time_bnds"

    coord = Coordinate.from_xarray(nemo_ocean_16, "time_counter")
    coord_dset = coord.to_xarray(encoding=False)
    assert "time_counter_bounds" in coord_dset
    assert "bounds" in coord_dset["time"].encoding
    assert coord_dset["time"].encoding["bounds"] == "time_counter_bounds"

    coord_dset = coord.to_xarray(encoding=True)
    assert "time_counter_bounds" in coord_dset
    assert "bounds" in coord_dset["time_counter"].encoding
    assert (
        coord_dset["time_counter"].encoding["bounds"] == "time_counter_bounds"
    )

    coord = Coordinate.from_xarray(nemo_ocean_16, "nav_lat")
    coord_dset = coord.to_xarray(encoding=False)
    assert "bounds_lat" in coord_dset
    assert "bounds" in coord_dset["latitude"].encoding
    assert coord_dset["latitude"].encoding["bounds"] == "bounds_lat"

    coord_dset = coord.to_xarray(encoding=True)
    assert "bounds_lat" in coord_dset
    assert "bounds" in coord_dset["nav_lat"].encoding
    assert coord_dset["nav_lat"].encoding["bounds"] == "bounds_lat"

    coord = Coordinate.from_xarray(nemo_ocean_16, "nav_lon")
    coord_dset = coord.to_xarray(encoding=False)
    assert "bounds_lon" in coord_dset
    assert "bounds" in coord_dset["longitude"].encoding
    assert coord_dset["longitude"].encoding["bounds"] == "bounds_lon"

    coord_dset = coord.to_xarray(encoding=True)
    assert "bounds_lon" in coord_dset
    assert "bounds" in coord_dset["nav_lon"].encoding
    assert coord_dset["nav_lon"].encoding["bounds"] == "bounds_lon"


def test_era5_check_if_independent_when_name_encoding_set(era5_netcdf):
    era5_netcdf["latitude"].encoding["name"] = "lat"
    coord = Coordinate.from_xarray(era5_netcdf, "latitude")
    assert coord.is_independent


def test_to_dict_store_proper_keys(nemo_ocean_16):
    details = Coordinate.from_xarray(nemo_ocean_16, "nav_lon").to_dict()
    assert isinstance(details, dict)
    assert "values" in details
    assert "units" in details
    assert "axis" in details


def test_to_dict_use_standard_name(nemo_ocean_16):
    details = Coordinate.from_xarray(nemo_ocean_16, "nav_lon").to_dict()
    assert isinstance(details["values"], list)
    assert details["units"] == "degrees_east"
    assert details["axis"] == "LONGITUDE"


def test_to_dict_store_all_values(nemo_ocean_16):
    details = Coordinate.from_xarray(nemo_ocean_16, "nav_lon").to_dict()
    assert isinstance(details["values"], list)
    assert np.all(np.array(details["values"]) == nemo_ocean_16.nav_lon.values)


def test_to_dict_store_unique_values(nemo_ocean_16):
    details = Coordinate.from_xarray(nemo_ocean_16, "nav_lon").to_dict(True)
    assert isinstance(details["values"], list)
    assert np.all(
        np.array(details["values"]) == np.unique(nemo_ocean_16.nav_lon.values)
    )
