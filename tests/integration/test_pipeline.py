"""End-to-end DataCube pipelines, mirroring the manual checks in development.py.

These exercise the chained operations (resample -> regrid / to_regular) and the
zarr round-trip at the DataCube level, on a small subset of the rotated-pole
downscaled-over-Italy file.
"""
import numpy as np
import pytest

from geokube.backend import open_datacube
from geokube.core.coord_system import GeogCS, RotatedGeogCS
from geokube.core.domain import Domain
from geokube.core.enums import RegridMethod

DOWNSCALED = "tests/resources/era5_downscaled_over_italy.nc"


@pytest.fixture
def downscaled_subset():
    kube = open_datacube(DOWNSCALED, decode_coords="all")
    # 24 hourly steps over central Italy -> small rotated grid (~50x38).
    return kube.geobbox(north=42.5, south=41.5, west=12.0, east=13.0)


@pytest.mark.integration
def test_pipeline_resample_then_regrid_then_zarr(downscaled_subset, tmp_path):
    lat = np.arange(41.5, 42.5, 0.1)
    lon = np.arange(12.0, 13.0, 0.1)
    target = Domain(
        coords={"latitude": lat, "longitude": lon}, crs=GeogCS(6371229)
    )

    result = downscaled_subset.resample(
        frequency="6h", operator="mean"
    ).regrid(target, method=RegridMethod.BILINEAR)

    # resample collapsed 24 hourly steps into 4 six-hourly means...
    assert result.time.size == 4
    # ...and regrid moved it onto the regular target grid.
    assert not isinstance(result.domain.crs, RotatedGeogCS)
    field = next(iter(result.fields.values()))
    assert np.allclose(field.latitude.values, lat)
    assert np.allclose(field.longitude.values, lon)
    assert 0 not in field.shape

    # The whole pipeline result survives a zarr round-trip.
    path = str(tmp_path / "pipeline.zarr")
    result.to_zarr(path, mode="w", consolidated=True)
    reopened = open_datacube(path, decode_coords="all", chunks={})
    back = next(iter(reopened.fields.values()))
    np.testing.assert_allclose(
        np.asarray(back.values), np.asarray(field.values), equal_nan=True
    )


@pytest.mark.integration
def test_pipeline_resample_then_to_regular(downscaled_subset):
    result = downscaled_subset.resample(
        frequency="6h", operator="mean"
    ).to_regular()

    assert result.time.size == 4
    assert not isinstance(result.domain.crs, RotatedGeogCS)
    field = next(iter(result.fields.values()))
    assert field.latitude.type.name == "INDEPENDENT"
    assert field.longitude.type.name == "INDEPENDENT"
    assert 0 not in field.shape
