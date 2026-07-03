import numpy as np

from geokube.backend import open_datacube
from geokube.core.coord_system import RotatedGeogCS

DOWNSCALED = "tests/resources/era5_downscaled_over_italy.nc"


def _subset():
    # A small rotated-pole DataCube cut from the downscaled-over-Italy file,
    # so the zarr round-trip stays fast.
    kube = open_datacube(DOWNSCALED, decode_coords="all")
    return kube.geobbox(north=42.5, south=41.5, west=12.0, east=13.0)


def test_datacube_to_zarr_roundtrip(tmp_path):
    sub = _subset()
    path = str(tmp_path / "roundtrip.zarr")

    # Default encoding — the supported persistence path, as in development.py.
    sub.to_zarr(path, mode="w", consolidated=True)

    reopened = open_datacube(path, decode_coords="all", chunks={})

    assert set(reopened.fields) == set(sub.fields)
    for name, field in sub.fields.items():
        np.testing.assert_allclose(
            np.asarray(reopened[name].values),
            np.asarray(field.values),
            equal_nan=True,
        )

    # The rotated-pole CRS must survive the zarr round-trip.
    assert isinstance(reopened.domain.crs, RotatedGeogCS)
