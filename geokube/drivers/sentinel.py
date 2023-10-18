from glob import iglob
import os
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS, GeographicCRS
import xarray as xr

from geokube import CoordinateSystem, Grid, GridField, Projection, axis


def open_sentinel(path: str) -> pd.DataFrame:
    path_time = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
    time = pd.to_datetime([path_time]).to_numpy()
    data: list[tuple[str, str, GridField]] = []
    coords: dict[str, dict[axis.Axis, np.ndarray]] = {}
    attrs: dict[str, Any]
    crs: CRS
    transformer: Transformer
    coord_system: CoordinateSystem
    path_like = os.path.join(
        path, 'GRANULE', '*', 'IMG_DATA', '*', '*_B*_*.jp2'
    )
    for i, full_path in enumerate(iglob(path_like)):
        # Resolve the parameters.
        res, file = full_path.split(os.sep)[-2:]
        # time, band = file.split('_')[-3:-1]
        band = file.split('_')[-2]

        # Open dataset.
        dset = xr.open_dataset(full_path)

        # Set the attributes, CRS, coordinate system, and transformer only when
        # the first dataset id opened.
        if not i:
            attrs = dset.attrs
            crs = CRS.from_cf(dset['spatial_ref'].attrs)
            coord_system = CoordinateSystem(
                horizontal=Projection(crs=crs), time=axis.time
            )
            transformer = Transformer.from_crs(
                crs_from=crs, crs_to=GeographicCRS(), always_xy=True
            )

        # Prepare coordinates once for each resolution.
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
            dim_axes=(axis.time, axis.y, axis.x),
            properties=attrs
        )

        # Append the result.
        data.append((res, band, field))

    return pd.DataFrame(data=data, columns=['resolutions', 'bands', 'fields'])
