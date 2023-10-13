import pint
import xarray as xr

from geokube import axis, CoordinateSystem, CRS, Geodetic, RotatedGeodetic
from geokube.core import domain, field


_FEATURE_MAP: dict[str | None, type[field.Field]] = {
    'point': field.PointsField,
    # 'timeSeries': ...,
    # 'trajectory': ...,
    'profile': field.ProfileField,
    # 'timeSeriesProfile': ...,
    # 'trajectoryProfile': ...,
    None: field.GridField
}


def open_cf_netcdf(path: str, variable: str, **kwargs) -> field.Field:
    var_name = str(variable)
    kwargs.setdefault('decode_coords', 'all')
    with xr.open_mfdataset(path, **kwargs) as dset:
        dset_cf = dset.cf
        dset_coords = dict(dset.coords)

        # Horizontal coordinate system:
        if gmn := dset_cf.grid_mapping_names:
            crs_var_name = next(iter(gmn.values()))[0]
            hor_crs = CRS.from_cf(dset[crs_var_name].attrs)
            dset_coords.pop(crs_var_name)
        else:
            hor_crs = Geodetic()

        # Coordinates.
        coords = {}
        for cf_coord, cf_coord_names in dset_cf.coordinates.items():
            assert len(cf_coord_names) == 1
            cf_coord_name = cf_coord_names[0]
            coord = dset_coords.pop(cf_coord_name)
            axis_ = axis._from_string(cf_coord)
            coords[axis_] = pint.Quantity(
                coord.to_numpy(), coord.attrs.get('units')
            )
        for cf_axis, cf_axis_names in dset_cf.axes.items():
            assert len(cf_axis_names) == 1
            cf_axis_name = cf_axis_names[0]
            if cf_axis_name in dset_coords:
                coord = dset_coords.pop(cf_axis_name)
                axis_ = axis._from_string(cf_axis.lower())
                if isinstance(hor_crs, RotatedGeodetic):
                    if axis_ is axis.x:
                        axis_ = axis.grid_longitude
                    elif axis_ is axis.y:
                        axis_ = axis.grid_latitude
                coords[axis_] = pint.Quantity(
                    coord.to_numpy(), coord.attrs.get('units')
                )

        # Coordinate system.
        time = {
            axis_
            for axis_ in coords
            if isinstance(axis_, axis.Time) and coords[axis_].ndim
        }
        assert len(time) <= 1
        elev = {
            axis_
            for axis_ in coords
            if isinstance(axis_, axis.Elevation) and coords[axis_].ndim
        }
        assert len(elev) <= 1
        # TODO: Consider user axes.
        # TODO: Consider multiple vertical axes (e.g. heights at 2 and 10 m).
        crs = CoordinateSystem(
            horizontal=hor_crs,
            elevation=elev.pop() if elev else None,
            time=time.pop() if time else None
        )

        # Domain.
        field_type = _FEATURE_MAP[dset.attrs.get('featureType')]
        domain_type = field_type._DOMAIN_TYPE
        domain_ = domain_type(coords=coords, coord_system=crs)

        # Data.
        var = dset[var_name]
        data = pint.Quantity(var.data, var.attrs.get('units'))

        # Field
        field = field_type(name=var_name, domain=domain_, data=data)
        return field
