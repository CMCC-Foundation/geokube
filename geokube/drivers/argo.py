import pint
import xarray as xr

from geokube import (
    axis, CoordinateSystem, GeographicCRS, Profile, ProfileField, units
)


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


def open_argo(path: str, variable: str, **kwargs) -> ProfileField:
    import gsw

    var = str(variable)
    redundant_vars = set(_ARGO_DATA_VARS) - {'PRES', var}
    kwa = {'decode_coords': 'all', 'drop_variables': redundant_vars}
    with xr.open_dataset(path, **kwa, **kwargs) as dset:
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
        domain = Profile(coords=coords, coord_system=coord_system)
        data_var = dset[var]
        data_vals, data_units = data_var.data, data_var.attrs.get('units', '')
        field = ProfileField(
            name=var.lower(),
            domain=domain,
            data=pint.Quantity(data_vals, data_units),
            anciliary=None,
            properties=dset.attrs,
            encoding=dset.encoding
        )
        return field
