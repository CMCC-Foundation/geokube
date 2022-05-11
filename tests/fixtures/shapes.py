import geopandas as gpd
import pytest


@pytest.fixture
def it_shape_100_km():
    path = "tests//resources//italy_shapefile//it_100km.shp"
    yield gpd.read_file(path).to_crs(epsg=4326)["geometry"]
