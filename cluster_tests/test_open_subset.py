import pytest
from dask.distributed import Client, LocalCluster
from geokube.backend.netcdf import open_datacube, open_dataset
from tests import TimeCounter
import xarray as xr
import numpy as np
import cluster_tests.datasets_conf as DS

RES_PATH = "tests/resources/res.nc"


@pytest.fixture
def client():
    cluster = LocalCluster(
        interface="ib0",
        dashboard_address=":9992",
        local_directory="/work/asc/jw25619/_workers",
    )
    client = Client(cluster, asynchronous=False)
    yield client
    client.close()
    cluster.close()


def test_1(client):
    def f():
        path = DS.ERA5
        hc = open_datacube(
            path, decode_coords="all", parallel=True, chunks=DS.ERA5_CHUNKS
        )["tp"]
        res = hc.geobbox(
            north=90, south=85, west=-20, east=20, roll_if_needed=True
        ).sel(time="2018-01-27")
        res.to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert np.all(ds.time.dt.day == 27)
    assert np.all(ds.time.dt.month == 1)
    assert np.all(ds.time.dt.year == 2018)
    assert np.all(ds.longitude <= 20)
    assert np.all(ds.longitude >= -20)
    assert np.any(ds.longitude < -1)
    assert np.all(ds.latitude <= 90)
    assert np.all(ds.latitude >= 85)


def test_2(client):
    def f():
        path = DS.ERA5
        hc = open_datacube(
            path, decode_coords="all", parallel=True, chunks=DS.ERA5_CHUNKS
        )["tp"]
        res = hc.geobbox(north=90, south=85, west=10, east=35, roll_if_needed=True)
        # res = DataCube.from_xarray(res)
        res = res.sel({"time": {"year": [2010, 2012], "month": "02", "day": [1, 4, 7]}})
        res.to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert np.all((ds.time.dt.day == 1) | (ds.time.dt.day == 4) | (ds.time.dt.day == 7))
    assert np.all((ds.time.dt.year == 2010) | (ds.time.dt.year == 2012))
    assert np.all(ds.time.dt.month == 2)
    assert np.all(ds.longitude >= 10)
    assert np.any(ds.longitude <= 35)
    assert np.all(ds.latitude <= 90)
    assert np.all(ds.latitude >= 85)


def test_3(client):
    def f():
        path = DS.E_OBS
        hc = open_dataset(
            path,
            pattern=DS.E_OBS_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.E_OBS_CHUNKS,
        )["pp"]
        res = hc.geobbox(north=60, south=55, west=-5, east=5).cubes[0]
        res = res.sel({"time": {"year": [2010, 2012], "month": 2, "day": [16, 4, 7]}})
        res.to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert np.all(
        (ds.time.dt.day == 16) | (ds.time.dt.day == 4) | (ds.time.dt.day == 7)
    )
    assert np.all((ds.time.dt.year == 2010) | (ds.time.dt.year == 2012))
    assert np.all(ds.time.dt.month == 2)
    assert np.all(ds.latitude >= 55)
    assert np.any(ds.latitude <= 60)
    assert np.all(ds.longitude <= 5)
    assert np.all(ds.longitude >= -5)


def test_4(client):
    def f():
        path = DS.E_OBS
        hc = open_dataset(
            path,
            pattern=DS.E_OBS_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.E_OBS_CHUNKS,
        )["pp"]
        res = hc.locations(latitude=60, longitude=44).cubes[0]
        res = res.sel({"time": {"year": [2011, 2012], "month": 2, "day": [16, 20]}})
        res.to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert np.all((ds.time.dt.day == 16) | (ds.time.dt.day == 20))
    assert np.all((ds.time.dt.year == 2011) | (ds.time.dt.year == 2012))
    assert np.all(ds.time.dt.month == 2)


def test_5(client):
    def f():
        path = DS.E_OBS
        hc = open_dataset(
            path,
            pattern=DS.E_OBS_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.E_OBS_CHUNKS,
        )["pp"]
        res = hc.locations(latitude=[60, 44], longitude=[44, 48]).cubes[0]
        res = res.sel({"time": {"year": [2010, 2012], "month": 2, "day": [16, 4, 7]}})
        res.to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert np.all(
        (ds.time.dt.day == 16) | (ds.time.dt.day == 4) | (ds.time.dt.day == 7)
    )
    assert np.all((ds.time.dt.year == 2010) | (ds.time.dt.year == 2012))
    assert np.all(ds.time.dt.month == 2)
    assert np.all((ds.latitude.values - np.array([60, 44])) <= 0.1)
    assert np.all((ds.longitude.values - np.array([44, 48])) <= 0.1)


def test_6(client):
    def f():
        path = DS.E_OBS
        hc = open_dataset(
            path,
            pattern=DS.E_OBS_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.E_OBS_CHUNKS,
        )["pp"]
        res = hc.geobbox(north=60, south=55, west=350, east=360).cubes[0]
        res = res.sel({"time": {"year": [2010, 2012], "month": 2, "day": [16, 4, 7]}})
        res.to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert np.all(
        (ds.time.dt.day == 16) | (ds.time.dt.day == 4) | (ds.time.dt.day == 7)
    )
    assert np.all((ds.time.dt.year == 2010) | (ds.time.dt.year == 2012))
    assert np.all(ds.time.dt.month == 2)
    assert np.all(ds.latitude >= 55)
    assert np.any(ds.latitude <= 60)
    assert np.all(ds.longitude <= 360)
    assert np.all(ds.longitude >= 350)


def test_7(client):
    def f():
        path = DS.NEMO_GLOBAL
        hc = open_dataset(
            path,
            pattern=DS.NEMO_GLOBAL_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.NEMO_GLOBAL_CHUNKS,
        )["so"]
        res = hc.geobbox(
            north=30, south=-10, west=20, east=60, roll_if_needed=False
        ).sel(depth=slice(300, 1000))
        res.cubes[0].to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert ds.crs.grid_mapping_name == "curvilinear_grid"
    assert (np.sum(ds.nav_lat.values <= 30) / ds.nav_lat.values.size) > 0.95
    assert (np.sum(ds.nav_lat.values >= -10) / ds.nav_lat.values.size) > 0.95
    assert (np.sum(ds.nav_lon.values <= 60) / ds.nav_lon.values.size) > 0.95
    assert (np.sum(ds.nav_lon.values >= 20) / ds.nav_lon.values.size) > 0.95
    assert np.all(ds.deptht <= 1000)
    assert np.all(ds.deptht >= 300)


def test_8(client):
    def f():
        path = DS.NEMO_GLOBAL
        hc = open_dataset(
            path,
            pattern=DS.NEMO_GLOBAL_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.NEMO_GLOBAL_CHUNKS,
        )["so"]
        res = hc.locations(latitude=[-20, 15, 35], longitude=150)
        res.cubes[0].to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert ds.crs.grid_mapping_name == "curvilinear_grid"
    assert len(ds.points) == 3
    assert np.all((ds.nav_lat.values - np.array([-20, 15, 35])) <= 0.1)
    assert np.all((ds.nav_lon.values - np.array([150])) <= 0.1)


def test_9(client):
    def f():
        path = DS.NEMO_GLOBAL
        hc = open_dataset(
            path,
            pattern=DS.NEMO_GLOBAL_PATTERN,
            decode_coords="all",
            parallel=True,
            chunks=DS.NEMO_GLOBAL_CHUNKS,
            mapping={"so": {"api": "some_so"}},
        )["some_so"]
        res = hc.locations(latitude=[-20, 15, 35], longitude=150)
        res.cubes[0].to_xarray().to_netcdf(RES_PATH)

    with TimeCounter(print=True, log=True) as tc:
        _ = client.submit(f).result()

    ds = xr.open_dataset(RES_PATH)
    assert ds.crs.grid_mapping_name == "curvilinear_grid"
    assert len(ds.points) == 3
    assert np.all((ds.nav_lat.values - np.array([-20, 15, 35])) <= 0.1)
    assert np.all((ds.nav_lon.values - np.array([150])) <= 0.1)
