import pytest
import xarray as xr


@pytest.fixture
def era5_point_domain():
    return xr.open_mfdataset(
        "tests//resources//point_domain*.nc", chunks="auto", decode_coords="all"
    )


@pytest.fixture
def era5_netcdf():
    yield xr.open_mfdataset(
        "tests//resources//era5-single*.nc", chunks="auto", decode_coords="all"
    )


@pytest.fixture
def era5_globe_netcdf():
    yield xr.open_dataset(
        "tests//resources//globe-era5-single-levels-reanalysis.nc",
        chunks="auto",
        decode_coords="all",
    )


@pytest.fixture
def era5_rotated_netcdf_tmin2m():
    yield xr.open_mfdataset(
        "tests//resources//rlat-rlon-tmin2m.nc", chunks="auto", decode_coords="all"
    )


@pytest.fixture
def era5_rotated_netcdf_wso():
    yield xr.open_mfdataset(
        "tests//resources//rlat-rlon-wso.nc", chunks="auto", decode_coords="all"
    )


@pytest.fixture
def era5_rotated_netcdf():
    yield xr.open_mfdataset(
        "tests//resources//rlat-*.nc", chunks="auto", decode_coords="all"
    )


@pytest.fixture
def era5_rotated_netcdf_lat(era5_rotated_netcdf_wso):
    yield era5_rotated_netcdf_wso.lat


@pytest.fixture
def era5_rotated_netcdf_soil(era5_rotated_netcdf_wso):
    yield era5_rotated_netcdf_wso.soil1


@pytest.fixture
def era5_rotated_netcdf_soil_bnds(era5_rotated_netcdf_wso):
    yield era5_rotated_netcdf_wso.soil1_bnds


@pytest.fixture
def nemo_ocean_16():
    yield xr.open_mfdataset(
        "tests//resources//nemo_ocean_16.nc", chunks="auto", decode_coords="all"
    )
