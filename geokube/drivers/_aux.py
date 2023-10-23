import pint
import xarray as xr

from geokube import (
    axis,
    CoordinateSystem,
    Geodetic,
    Grid,
    GridField,
    Profiles,
    ProfilesField,
    RotatedGeodetic,
    units
)


# ERA5 ------------------------------------------------------------------------

def open_era5(path: str, variable: str, **kwargs) -> GridField:
    var = str(variable)
    with xr.open_dataset(path, decode_coords='all', **kwargs) as dset:
        time = dset['time']
        time_vals = time.to_numpy()
        lat = dset['latitude']
        lat_vals, lat_units = lat.to_numpy(), lat.attrs['units']
        lon = dset['longitude']
        lon_vals, lon_units = lon.to_numpy(), lon.attrs['units']
        coords = {
            axis.time: pint.Quantity(time_vals),
            axis.latitude: pint.Quantity(lat_vals, lat_units),
            axis.longitude: pint.Quantity(lon_vals, lon_units)
        }
        coord_system = CoordinateSystem(horizontal=Geodetic(), time=axis.time)
        domain = Grid(coords=coords, coord_system=coord_system)
        data_var = dset[var]
        data_vals, data_units = data_var.data, data_var.attrs.get('units', '')
        field = GridField(
            name=var,
            domain=domain,
            data=pint.Quantity(data_vals, data_units),
            anciliary=None,
            properties=dset.attrs,
            encoding=dset.encoding
        )
        return field


def open_era5_it(path: str, variable: str, **kwargs) -> GridField:
    # https://dds.cmcc.it/#/dataset/era5-downscaled-over-italy/hourly

    # TODO: Consider time bounds.
    var = str(variable)
    with xr.open_dataset(path, decode_coords='all', **kwargs) as dset:
        time = dset['time']
        time_vals = time.to_numpy()
        y = dset['rlat']
        y_vals, y_units = y.to_numpy(), y.attrs['units']
        x = dset['rlon']
        x_vals, x_units = x.to_numpy(), x.attrs['units']
        lat = dset['lat']
        lat_vals, lat_units = lat.to_numpy(), lat.attrs['units']
        lon = dset['lon']
        lon_vals, lon_units = lon.to_numpy(), lon.attrs['units']
        coords = {
            axis.time: pint.Quantity(time_vals),
            axis.grid_latitude: pint.Quantity(y_vals, y_units),
            axis.grid_longitude: pint.Quantity(x_vals, x_units),
            axis.latitude: pint.Quantity(lat_vals, lat_units),
            axis.longitude: pint.Quantity(lon_vals, lon_units)
        }
        crs_attrs = dset['crs'].attrs
        hor_crs = RotatedGeodetic(
            pole_longitude=crs_attrs['grid_north_pole_longitude'],
            pole_latitude=crs_attrs['grid_north_pole_latitude'],
        )
        coord_system = CoordinateSystem(horizontal=hor_crs, time=axis.time)
        domain = Grid(coords=coords, coord_system=coord_system)
        data_var = dset[var]
        data_vals, data_units = data_var.data, data_var.attrs.get('units', '')
        # NOTE: The order of the dimension axes corresponds to the default.
        field = GridField(
            name=var,
            domain=domain,
            data=pint.Quantity(data_vals, data_units),
            anciliary=None,
            properties=dset.attrs,
            encoding=dset.encoding
        )
        return field


# E-OBS -----------------------------------------------------------------------

def open_eobs(path: str, variable: str, **kwargs) -> GridField:
    var = str(variable)
    with xr.open_dataset(path, decode_coords='all', **kwargs) as dset:
        time = dset['time']
        time_vals = time.to_numpy()
        lat = dset['latitude']
        lat_vals, lat_units = lat.to_numpy(), lat.attrs['units']
        lon = dset['longitude']
        lon_vals, lon_units = lon.to_numpy(), lon.attrs['units']
        coords = {
            axis.time: pint.Quantity(time_vals),
            axis.latitude: pint.Quantity(lat_vals, lat_units),
            axis.longitude: pint.Quantity(lon_vals, lon_units)
        }
        coord_system = CoordinateSystem(horizontal=Geodetic(), time=axis.time)
        domain = Grid(coords=coords, coord_system=coord_system)
        data_var = dset[var]
        data_vals, data_units = data_var.data, data_var.attrs.get('units', '')
        field = GridField(
            name=var,
            domain=domain,
            data=pint.Quantity(data_vals, data_units),
            anciliary=None,
            properties=dset.attrs,
            encoding=dset.encoding
        )
        return field
