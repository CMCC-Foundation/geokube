from collections.abc import Sequence
from glob import iglob
import os
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import GeographicCRS
import xarray as xr

from geokube import Collection, CoordinateSystem, Cube, Grid, GridField, axis
from geokube.core.crs import CRS, TransverseMercatorProjection


# TODO: Check whether the units of X and Y in the sentinel data correspond to
# the default units for X and Y axes (meters).
# TODO: Chenge the name of `open` because it is a name of a built-in function.


def open(
    path: str,
    resolutions: Sequence[str] | None = None,
    bands: Sequence[str] | None = None,
    xarray_kwargs: dict[str, Any] = None,
    **kwargs
) -> Collection:
    # Extract the time from the path.
    path_time = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
    time = pd.to_datetime([path_time]).to_numpy()

    # Extract the resolutions, bands, and paths.
    path_like = os.path.join(
        path, 'GRANULE', '*', 'IMG_DATA', '*', '*_B*_*.jp2'
    )
    tmp_data: list[tuple[str, str, str]] = []
    for i, full_path in enumerate(iglob(path_like)):
        # Resolve the parameters.
        res, file = full_path.split(os.sep)[-2:]
        # time, band = file.split('_')[-3:-1]
        band = file.split('_')[-2]
        tmp_data.append((res, band, full_path, None))

    # Prepare the data frame. The cubes are going to be added later.
    data = pd.DataFrame(
        data=tmp_data, columns=['resolutions', 'bands', 'paths', 'cubes']
    )
    mask = False
    if resolutions is not None:
        data = data[np.isin(data['resolutions'], resolutions)]
        mask = True
    if bands is not None:
        data = data[np.isin(data['bands'], bands)]
        mask = True
    if mask:
        data.reset_index(inplace=True, drop=True)

    # Creating the coordinate system (only once), coordinates (once for each)
    # resolution, fields, and cubes.
    transformer: Transformer
    coords: dict[str, dict[axis.Axis, np.ndarray]] = {}
    for i, res, band, full_path, _ in data.itertuples(index=True, name=None):
        dset = xr.open_dataset(full_path, **(xarray_kwargs or {}))

        # Set the attributes, CRS, coordinate system, and transformer only when
        # the first dataset is opened.
        if not i:
            attrs = dset.attrs
            crs = CRS.from_cf(dset['spatial_ref'].attrs)._crs
            proj = object.__new__(TransverseMercatorProjection)
            CRS.__init__(proj, crs=crs)
            coord_system = CoordinateSystem(horizontal=proj, time=axis.time)
            transformer = Transformer.from_crs(
                crs_from=crs, crs_to=GeographicCRS(), always_xy=True
            )

        # Prepare the coordinates once for each resolution.
        res_coords = coords.get(res)
        if res_coords is None:
            x_vals, y_vals = dset['x'].to_numpy(), dset['y'].to_numpy()
            lon_vals, lat_vals = transformer.transform(
                *np.meshgrid(x_vals, y_vals)
            )
            coords[res] = res_coords = {
                axis.time: time,
                axis.y: y_vals,
                axis.x: x_vals,
                axis.latitude: lat_vals,
                axis.longitude: lon_vals
            }

        # Create field.
        field = GridField(
            name=f'{res}_{band}',
            domain=Grid(coords=res_coords, coord_system=coord_system),
            data=dset['band_data'].data,
            properties=attrs
        )

        # Create cube and add it to the data frame.
        data.iat[i, -1] = Cube(fields=[field])

    # Creating and returning the collection of cubes.
    del data['paths']
    return Collection(data=data)
