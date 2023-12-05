import pandas as pd
import xarray as xr
import cf_xarray as cfxr  # pylint: disable=unused-import  # noqa: F401

from geokube import CoordinateSystem, CRS, Geodetic, RotatedGeodetic, axis
from geokube.core import domain, field


_FEATURE_MAP: dict[str | None, type[field.Field]] = {
    'point': field.PointsField,
    # 'timeSeries': ...,
    # 'trajectory': ...,
    'profile': field.ProfilesField,
    # 'timeSeriesProfile': ...,
    # 'trajectoryProfile': ...,
    None: field.GridField
}


def open_cf_netcdf(path: str, variable: str, **kwargs) -> field.Field:
    var_name = str(variable)
    kwargs.setdefault('decode_coords', 'all')
    with xr.open_dataset(path, **kwargs) as dset:
        dset_cf = dset.cf
        dset_coords = dict(dset.coords)

        # Horizontal coordinate system. ---------------------------------------
        if gmn := dset_cf.grid_mapping_names:
            crs_var_name = next(iter(gmn.values()))[0]
            hor_crs = CRS.from_cf(dset[crs_var_name].attrs)
            dset_coords.pop(crs_var_name)
        else:
            hor_crs = Geodetic()

        # Axis names. ---------------------------------------------------------
        axes = {}
        for standard_name, name in dset_cf.standard_names.items():
            assert len(name) == 1
            axes[name[0]] = axis._from_string(standard_name)

        # Coordinates. --------------------------------------------------------
        coords = {}
        for cf_coord, cf_coord_names in dset_cf.coordinates.items():
            assert len(cf_coord_names) == 1
            cf_coord_name = cf_coord_names[0]
            coord = dset_coords.pop(cf_coord_name)
            coord.attrs.setdefault('units', 'dimensionless')
            axis_ = axis._from_string(cf_coord)
            coords[axis_] = xr.Variable(
                dims=tuple(axes.get(dim, dim) for dim in coord.dims),
                data=coord.to_numpy(),
                attrs=coord.attrs,
                encoding=coord.encoding
            )
        for cf_axis, cf_axis_names in dset_cf.axes.items():
            assert len(cf_axis_names) == 1
            cf_axis_name = cf_axis_names[0]
            if cf_axis_name in dset_coords:
                coord = dset_coords.pop(cf_axis_name)
                coord.attrs.setdefault('units', 'dimensionless')
                axis_ = axis._from_string(cf_axis.lower())
                if isinstance(hor_crs, RotatedGeodetic):
                    if axis_ is axis.x:
                        axis_ = axis.grid_longitude
                    elif axis_ is axis.y:
                        axis_ = axis.grid_latitude
                coords[axis_] = xr.Variable(
                    dims=tuple(axes.get(dim, dim) for dim in coord.dims),
                    data=coord.to_numpy(),
                    attrs=coord.attrs,
                    encoding=coord.encoding
                )
        # Time bounds.
        if t_bnds := (dset.cf.bounds.get('time') or dset.cf.bounds.get('T')):
            assert len(t_bnds) == 1
            time_vals = dset[t_bnds[0]].to_numpy()
            time_bnds = pd.IntervalIndex.from_arrays(
                left=time_vals[:, 0], right=time_vals[:, 1], closed='both'
            )
            time_coord = coords[axis.time]
            coords[axis.time] = xr.Variable(
                dims=time_coord.dims,
                data=time_bnds,
                attrs=time_coord.attrs,
                encoding=time_coord.encoding
            )

        # Coordinate system. --------------------------------------------------
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

        # Domain. -------------------------------------------------------------
        result_dset = domain.Domain.as_xarray_dataset(coords, crs)
        result_dset = result_dset.drop_indexes(result_dset.xindexes)
        result_dset.encoding.update(dset.encoding)

        # Data. ---------------------------------------------------------------
        var = dset[var_name]
        data = xr.Variable(
            dims=tuple(axes.get(dim, dim) for dim in var.dims),
            data=var.to_numpy(),
            attrs=var.attrs,
            encoding=var.encoding
        )
        data.attrs['grid_mapping'] = result_dset.attrs['grid_mapping']
        data.attrs.setdefault('units', 'dimensionless')
        data_vars = {var_name: data}
        result_dset = result_dset.assign(data_vars)
        result_dset.attrs[field._FIELD_NAME_ATTR_] = var_name

        # Field. --------------------------------------------------------------
        field_type = _FEATURE_MAP[dset.attrs.get('featureType')]
        result = field_type._from_xarray_dataset(result_dset)
        return result
