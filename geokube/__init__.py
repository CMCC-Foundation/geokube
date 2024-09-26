import geokube.backend
from geokube import version

open_dataset = geokube.backend.open_dataset
open_datacube = geokube.backend.open_datacube

from geokube.core.coord_system import (
    AlbersEqualArea,
    GeogCS,
    Geostationary,
    LambertAzimuthalEqualArea,
    LambertConformal,
    Mercator,
    Orthographic,
    RegularLatLon,
    RotatedGeogCS,
    Stereographic,
    TransverseMercator,
    VerticalPerspective,
    CurvilinearGrid,
)

__version__ = version.__version__