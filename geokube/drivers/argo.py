from collections.abc import Sequence
from typing import Any

import pint
import xarray as xr

from geokube import (
    CoordinateSystem, Cube, Geodetic, Profiles, ProfilesField, axis, units
)


# TODO: Chenge the name of `open` because it is a name of a built-in function.


_ARGO_DATA_VARS = (
    'TIME_QC',
    'POSITION_QC',
    'DC_REFERENCE',
    'DIRECTION',
    'VERTICAL_SAMPLING_SCHEME',
    'PRES',
    'PRES_QC',
    'PRES_ADJUSTED',
    'PRES_ADJUSTED_QC',
    'TEMP',
    'TEMP_QC',
    'TEMP_ADJUSTED',
    'TEMP_ADJUSTED_QC',
    'PSAL',
    'PSAL_QC',
    'PSAL_ADJUSTED',
    'PSAL_ADJUSTED_QC'
)


def open(
    path: str,
    variables: str | Sequence[str] | None = None,
    xarray_kwargs: dict[str, Any] | None = None,
    **kwargs
) -> ProfilesField | Cube:
    import gsw

    match variables:
        case None:
            vars_ = ['PRES', 'TEMP', 'PSAL']
        case str():
            vars_ = [variables]
        case Sequence():
            vars_ = list(variables)

    redundant_vars = set(_ARGO_DATA_VARS) - {'PRES', *vars_}
    kwa = {'decode_coords': 'all', 'drop_variables': redundant_vars}

    if xarray_kwargs:
        kwa |= xarray_kwargs

    with xr.open_dataset(path, **kwa) as dset:
        time = dset['TIME']
        time_vals = time.to_numpy()
        lat = dset['LATITUDE']
        lat_vals, lat_units = lat.to_numpy(), lat.attrs['units']
        lon = dset['LONGITUDE']
        lon_vals, lon_units = lon.to_numpy(), lon.attrs['units']
        pres = dset['PRES']
        pres_vals, pres_units = pres.to_numpy(), pres.attrs['units']
        pres_vals = pint.Quantity(pres_vals, pres_units).to('dbar').magnitude
        vert_vals = gsw.z_from_p(p=pres_vals, lat=lat_vals.reshape(-1, 1))
        vert_units = units['m']
        coords = {
            axis.time: pint.Quantity(time_vals),
            axis.vertical: pint.Quantity(vert_vals, vert_units),
            axis.latitude: pint.Quantity(lat_vals, lat_units),
            axis.longitude: pint.Quantity(lon_vals, lon_units)
        }
        coord_system = CoordinateSystem(
            horizontal=Geodetic(), elevation=axis.vertical, time=axis.time
        )
        domain = Profiles(coords=coords, coord_system=coord_system)
        fields = []
        for var in vars_:
            data_var = dset[var]
            data_vals = data_var.data
            data_units = data_var.attrs.get('units', '')
            try:
                data_qty = pint.Quantity(data_vals, data_units)
            except ValueError:
                # TODO: Consider how to deal with the "units" of `PSAL`.
                data_qty = pint.Quantity(data_vals)
            field = ProfilesField(
                name=var.lower(),
                domain=domain,
                data=data_qty,
                ancillary=None,
                properties=dset.attrs,
                encoding=dset.encoding
            )
            fields.append(field)
        return fields[0] if len(fields) == 1 else Cube(fields=fields)
