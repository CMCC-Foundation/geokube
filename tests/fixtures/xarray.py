import pytest
import xarray as xr

from geokube.backend import open_dataset
from geokube.core.datacube import DataCube


@pytest.fixture
def dataset():
    yield open_dataset(
        "tests//resources//*-single-levels-reanalysis_*",
        pattern=(
            "tests//resources//{dataset}-single-levels-reanalysis_{vars}.nc"
        ),
    )


@pytest.fixture
def dataset_rotated():
    yield open_dataset(
        "tests//resources//rlat-rlon-*",
        pattern="tests//resources//rlat-rlon-{vars}.nc",
    )


@pytest.fixture
def dataset_single_att():
    yield open_dataset(
        "tests//resources//*-single-levels-reanalysis_*",
        pattern="tests//resources//{dataset}-single-levels-reanalysis_{}.nc",
    )


@pytest.fixture
def dataset_idpattern():
    yield open_dataset(
        "tests//resources//era5-single-levels-reanalysis_*",
        pattern=(
            "tests//resources//{dataset}-single-levels-reanalysis_{vars}.nc"
        ),
        id_pattern="std_{units}",
    )


@pytest.fixture
def datacube_with_the_same_standard_name(era5_rotated_netcdf, era5_netcdf):
    era5_rotated_netcdf["TMIN_2M"].attrs["standard_name"] = "std_name1"
    era5_rotated_netcdf["W_SO"].attrs["standard_name"] = "std_name1"
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    yield dc


@pytest.fixture
def rotated_pole_datacube(era5_rotated_netcdf):
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    return dc


@pytest.fixture
def era5_point_domain():
    return xr.open_mfdataset(
        "tests//resources//point_domain*.nc",
        chunks="auto",
        decode_coords="all",
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
        "tests//resources//rlat-rlon-tmin2m.nc",
        chunks="auto",
        decode_coords="all",
    )


@pytest.fixture
def era5_rotated_netcdf_wso():
    yield xr.open_mfdataset(
        "tests//resources//rlat-rlon-wso.nc",
        chunks="auto",
        decode_coords="all",
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
    dset = xr.open_mfdataset(
        "tests//resources//nemo_ocean_16.nc",
        chunks="auto",
        decode_coords="all",
    )
    # NOTE: there are two time-related coordinates.
    # It is not supported yet to have multiple coordinates of the same AxisType
    dset["vt"].encoding["coordinates"] = "depthv nav_lat nav_lon"
    yield dset.drop(["time_centered", "time_centered_bounds"])
