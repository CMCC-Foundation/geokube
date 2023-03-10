import cf_units as cf
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geokube.core.axis import Axis, AxisType
from geokube.core.unit import Unit
from geokube.core.variable import Variable
from geokube.utils.attrs_encoding import CFAttributes
from tests.fixtures import *


def test_fails_on_wrong_type():
    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `number.Number`,"
            r" `numpy.ndarray`, `dask.array.Array`, or `xarray.Variable`*"
        ),
    ):
        _ = Variable({1, 2, 3, 4})

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `number.Number`,"
            r" `numpy.ndarray`, `dask.array.Array`, or `xarray.Variable`*"
        ),
    ):
        _ = Variable([1, 2, 3, 4])

    with pytest.raises(
        TypeError,
        match=(
            r"Expected argument is one of the following types `number.Number`,"
            r" `numpy.ndarray`, `dask.array.Array`, or `xarray.Variable`*"
        ),
    ):
        _ = Variable("some_data")


def test_init_proper_attrs_set():
    d = np.random.random((10, 50))
    v = Variable(
        data=d,
        dims=[AxisType.LATITUDE, AxisType.LONGITUDE],
        units="m",
        properties={"prop1": "aaa"},
        encoding={"enc1": "bbb"},
    )

    assert set(v.dim_names) == {"latitude", "longitude"}
    assert set(v.dim_ncvars) == {"latitude", "longitude"}
    assert v.properties == {"prop1": "aaa"}
    assert v.units == Unit("m")


def test_init_based_on_other_variable():
    d = np.random.random((10, 50))
    v = Variable(
        data=d,
        dims=[AxisType.LATITUDE, AxisType.LONGITUDE],
        units="m",
        properties={"prop1": "aaa"},
        encoding={"enc1": "bbb"},
    )
    v2 = Variable(data=v)
    assert id(v) != id(v2)
    assert np.all(v._data == v2._data)
    assert id(v._data) == id(v2._data)
    assert set(v2.dim_names) == {"latitude", "longitude"}
    assert set(v2.dim_ncvars) == {"latitude", "longitude"}
    assert v2.properties == {"prop1": "aaa"}
    assert v2.units == Unit("m")

    xrv = v2.to_xarray(encoding=True)
    assert isinstance(xrv, xr.Variable)
    assert xrv.attrs == dict(**v.properties, units="m")
    assert xrv.encoding == v.encoding
    assert np.all(xrv.values == d)
    assert set(xrv.dims) == {"latitude", "longitude"}

    v3 = Variable.from_xarray(xr.DataArray(xrv))
    assert id(v3) != id(v2)
    assert id(v3._data) == id(v2._data)
    assert set(v3.dim_names) == {"latitude", "longitude"}
    assert set(v3.dim_ncvars) == {"latitude", "longitude"}
    assert v3.properties == {"prop1": "aaa"}
    assert v3.units == Unit("m")


def test_init_proper_attr_set():
    d = np.random.random((10, 50, 20))
    v = Variable(
        data=d,
        dims=[
            Axis("lat", is_dim=True),
            Axis("lon", is_dim=True),
            Axis("depth", is_dim=True),
        ],
        units="m",
        properties={"prop1": "aaa"},
        encoding={"enc1": "bbb"},
    )
    assert set(v.dim_ncvars) == {"lat", "lon", "depth"}
    xrv = v.to_xarray(encoding=True)
    assert isinstance(xrv, xr.Variable)
    assert xrv.attrs == v.properties
    assert xrv.encoding == v.encoding
    assert np.all(xrv.values == d)
    assert set(xrv.dims) == {"lat", "lon", "depth"}

    xrv = v.to_xarray(encoding=False)
    assert isinstance(xrv, xr.Variable)
    assert xrv.attrs == v.properties
    assert xrv.encoding == v.encoding
    assert np.all(xrv.values == d)
    assert set(xrv.dims) == {"lat", "lon", "depth"}


def test_init_proper_attr_set_with_encoding_for_axes():
    d = da.random.random((10, 50, 20))
    v = Variable(
        data=d,
        dims=[
            Axis(
                name="lat",
                axistype=AxisType.LATITUDE,
                is_dim=True,
                encoding={"name": "latT"},
            ),
            Axis(
                name="lon",
                axistype=AxisType.LONGITUDE,
                is_dim=True,
                encoding={"name": "lonN"},
            ),
            Axis(
                name="depth",
                axistype=AxisType.VERTICAL,
                is_dim=True,
                encoding={"name": "depthH"},
            ),
        ],
        units="m",
        properties={"prop1": "aaa"},
        encoding={"enc1": "bbb"},
    )
    assert isinstance(v._data, da.Array)
    assert set(v.dim_ncvars) == {"latT", "lonN", "depthH"}
    assert set(v.dim_names) == {"lat", "lon", "depth"}
    assert v.units == Unit("m")

    xrv = v.to_xarray(encoding=True)
    assert isinstance(xrv, xr.Variable)
    assert xrv.attrs == v.properties
    assert xrv.encoding == v.encoding
    assert np.all(xrv.values == d)
    assert set(xrv.dims) == {"latT", "lonN", "depthH"}

    xrv = v.to_xarray(encoding=False)
    assert isinstance(xrv, xr.Variable)
    assert xrv.attrs == v.properties
    assert xrv.encoding == v.encoding
    assert np.all(xrv.values == d)
    assert set(xrv.dims) == {"lat", "lon", "depth"}
    assert xrv.attrs["units"] == "m"


def test_from_xarray_with_id_pattern(era5_netcdf):
    v = Variable.from_xarray(
        era5_netcdf["tp"], id_pattern="prefix:{units}_{long_name}"
    )

    d1 = v.dims[0]
    assert d1.type is AxisType.TIME
    assert (
        d1.name == "time"
    )  # if any id_pattern component is not found, then defaults is taken

    d2 = v.dims[1]
    assert d2.type is AxisType.LATITUDE
    assert d2.name == "prefix:degrees_north_latitude"
    assert d2.ncvar == "latitude"

    d3 = v.dims[2]
    assert d3.type is AxisType.LONGITUDE
    assert d3.name == "prefix:degrees_east_longitude"
    assert d3.ncvar == "longitude"

    xrv1 = v.to_xarray(encoding=True)
    assert xrv1.attrs == era5_netcdf["tp"].attrs
    assert xrv1.encoding == era5_netcdf["tp"].encoding
    assert set(xrv1.dims) == set(era5_netcdf["tp"].dims)

    xrv1 = v.to_xarray(encoding=False)
    assert xrv1.attrs == era5_netcdf["tp"].attrs
    assert xrv1.encoding == era5_netcdf["tp"].encoding
    assert set(xrv1.dims) == {
        "prefix:degrees_east_longitude",
        "prefix:degrees_north_latitude",
        "time",
    }


def test_from_xarray_rotated_pole_with_mapping(era5_rotated_netcdf_wso):
    v = Variable.from_xarray(
        era5_rotated_netcdf_wso["W_SO"],
        mapping={"soil1": {"name": "my_depth"}},
    )
    assert set(v.dim_ncvars) == (
        set(era5_rotated_netcdf_wso.dims.keys()) - {"bnds"}
    )
    mask = ~np.isnan(v._data)
    assert np.allclose(
        np.array(v._data)[mask],
        np.array(era5_rotated_netcdf_wso["W_SO"]._variable._data[mask]),
    )
    r1 = v.to_xarray(encoding=True)
    assert r1.dims == era5_rotated_netcdf_wso["W_SO"].dims
    assert r1.encoding == era5_rotated_netcdf_wso["W_SO"].encoding
    assert r1.attrs == era5_rotated_netcdf_wso["W_SO"].attrs

    r2 = v.to_xarray(encoding=False)
    assert set(r2.dims) == {
        "my_depth",
        "time",
        "grid_latitude",
        "grid_longitude",
    }
    assert r2.encoding == era5_rotated_netcdf_wso["W_SO"].encoding
    assert r2.attrs == era5_rotated_netcdf_wso["W_SO"].attrs
