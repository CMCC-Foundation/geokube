import cf_units as cf
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geokube.core.axis import Axis, Axis
from geokube.core.dimension import Dimension
from geokube.core.variable import Variable
from geokube.utils.attrs_encoding import CFAttributes
from tests.fixtures import *


def test_construct_from_numpy():
    d = np.random.random((10, 50))
    v = Variable(
        name="var1",
        data=d,
        units=cf.Unit("m"),
        dims=[
            Dimension("lat", Axis(atype=Axis.LATITUDE)),
            Dimension("lon", Axis(atype=Axis.LONGITUDE)),
        ],
    )
    assert v.name == "var1"
    assert v.units
    assert v.dims_names == ("lat", "lon")
    assert v.ndim == 2
    assert v.shape == (10, 50)
    assert v.dims[0].atype is Axis.LATITUDE
    assert v.dims[1].atype is Axis.LONGITUDE

    v = Variable(
        name="var1",
        data=d,
        units=cf.Unit("m"),
        dims=[
            Dimension("lat", Axis(atype=Axis.LATITUDE, name="lat")),
            Dimension("lon", Axis(atype=Axis.LONGITUDE)),
        ],
        cf_encoding={"standard_name": "variable_1"},
    )
    assert v.name == "var1"
    assert v.units
    assert v.dims_names == ("lat", "lon")
    assert v.ndim == 2
    assert v.shape == (10, 50)
    assert v.dims[0].atype is Axis.LATITUDE
    assert v.dims[1].atype is Axis.LONGITUDE
    xrv = v.to_xarray_variable()
    assert isinstance(xrv, xr.Variable)
    assert xrv.attrs["standard_name"] == "variable_1"
    assert np.all(xrv.values == d)
    assert xrv.dims == ("lat", "longitude")


def test_construct_from_dask():
    d = da.random.random((10, 50, 5))
    v = Variable(
        name="var2",
        data=d,
        units=cf.Unit("m"),
        dims=[
            Dimension("time", Axis(atype=Axis.TIME)),
            Dimension("lat", Axis(atype=Axis.LATITUDE)),
            Dimension("lon", Axis(atype=Axis.LONGITUDE)),
        ],
    )
    assert v.name == "var2"
    assert v.units
    assert v.dims_names == ("time", "lat", "lon")
    assert v.ndim == 3
    assert v.shape == (10, 50, 5)
    assert v.dims[0].atype is Axis.TIME
    assert v.dims[1].atype is Axis.LATITUDE
    assert v.dims[2].atype is Axis.LONGITUDE
    assert isinstance(v.data, da.Array)


def test_construct_from_xarray():
    d = da.random.random((10, 50, 5))
    xrv = xr.Variable(data=d, dims=("lat", "lon", "time"))
    v = Variable(
        name="var2",
        data=xrv,
        units="K",
        dims=[
            Dimension("time", Axis(atype=Axis.TIME)),
            Dimension("lat", Axis(atype=Axis.LATITUDE)),
            Dimension("lon", Axis(atype=Axis.LONGITUDE)),
        ],
    )
    assert v.name == "var2"
    assert v.units == cf.Unit("K")
    assert v.dims_names == ("time", "lat", "lon")
    assert v.ndim == 3
    assert v.shape == (10, 50, 5)
    assert v.dims[0].atype is Axis.TIME
    assert v.dims[1].atype is Axis.LATITUDE
    assert v.dims[2].atype is Axis.LONGITUDE


def test_convert_units(era5_netcdf):
    v = Variable.from_xarray_dataarray(era5_netcdf["tp"])
    v.convert_units(unit="cm")
    assert np.allclose(era5_netcdf["tp"].values * 100, v._variable.values)
    assert v.units == cf.Unit("cm")

    v = Variable.from_xarray_dataarray(era5_netcdf["tp"])
    assert v.units == cf.Unit("m")
    v2 = v.convert_units(unit="cm", inplace=False)
    assert id(v) != id(v2)
    assert v.units == cf.Unit("m")
    assert v2.units == cf.Unit("cm")
    assert np.allclose(v._variable.values, era5_netcdf["tp"])
    assert np.allclose(v2._variable.values, era5_netcdf["tp"] * 100)


def test_from_xarray_dataarray_1(
    era5_netcdf, era5_rotated_netcdf_tmin2m, nemo_ocean_16
):
    v = Variable.from_xarray_dataarray(era5_netcdf["longitude"])
    assert v.name == "longitude"
    assert v.dims_names == ("longitude",)
    xrv = v.to_xarray_variable()
    assert np.all(xrv == era5_netcdf["longitude"]._variable)

    v = Variable.from_xarray_dataarray(era5_netcdf["tp"])
    assert v.name == "tp"
    assert v.dims_names == ("time", "latitude", "longitude")
    assert v.units == cf.Unit("m")
    assert v.dims[0].atype is Axis.TIME
    assert v.dims[2].atype is Axis.LONGITUDE
    assert v.dims[1].atype is Axis.LATITUDE
    xrv = v.to_xarray_variable()
    assert np.all(xrv == era5_netcdf["tp"]._variable)

    v = Variable.from_xarray_dataarray(era5_netcdf["d2m"])
    assert v.name == "d2m"
    assert v.dims_names == ("time", "latitude", "longitude")
    assert v.units == cf.Unit("K")
    assert v.dims[0].atype is Axis.TIME
    assert v.dims[2].atype is Axis.LONGITUDE
    assert v.dims[1].atype is Axis.LATITUDE
    xrv = v.to_xarray_variable()
    assert np.all(xrv == era5_netcdf["d2m"]._variable)

    v = Variable.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["TMIN_2M"])
    assert v.name == "TMIN_2M"
    assert v._cf_encoding["standard_name"] == "air_temperature"
    assert v.units == cf.Unit("K")
    assert v.dims_names == ("time", "grid_latitude", "grid_longitude")
    assert v.dims[0].atype is Axis.TIME
    assert v.dims[0].axis.name == "time"
    assert v.dims[1].atype is Axis.Y
    assert v.dims[1].axis.name == "rlat"
    assert v.dims[2].atype is Axis.X
    assert v.dims[2].axis.name == "rlon"


def test_to_xarray_dataarray_1(era5_netcdf, era5_rotated_netcdf_tmin2m, nemo_ocean_16):
    v = Variable.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["TMIN_2M"])
    res = v.to_xarray_dataarray()
    assert set(res.coords.keys()) == {"time", "rlat", "rlon"}
    assert res["rlat"].shape == (259,)


def test_to_xarray_dataarray_2(nemo_ocean_16):
    v = Variable.from_xarray_dataarray(nemo_ocean_16["vt"])
    res = v.to_xarray_dataarray()
    assert set(res.coords.keys()) == {"time_counter", "depthv", "y", "x"}


def test_to_xarray_dataset(nemo_ocean_16):
    v = Variable.from_xarray_dataarray(nemo_ocean_16["vt"])
    res = v.to_xarray_dataset()
    assert isinstance(res, xr.Dataset)
    assert list(res.data_vars.keys()) == ["vt"]
    assert set(res.coords.keys()) == {"time_counter", "depthv", "y", "x"}


def test_split_attrs():
    attrs = {"standard_name": "aa", "some_text": "bb", "positive": "up"}
    properties, cf_encoding = CFAttributes.split_attrs(attrs)
    assert "standard_name" in cf_encoding
    assert "positive" in cf_encoding
    assert "some_text" in properties
    assert len(properties) == 1
    assert len(cf_encoding) == 2


# def test_variable_getitem(era5_netcdf):
#     v = Variable.from_xarray_dataarray(era5_netcdf["tp"]).to_xarray_variable()
#     import pdb;pdb.set_trace()
#     pass
