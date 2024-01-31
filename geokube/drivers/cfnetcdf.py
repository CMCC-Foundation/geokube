"""The driver for CF-compliant NetCDF data."""

from collections.abc import Sequence
import os
from typing import Any

import pandas as pd
import xarray as xr
import cf_xarray as cfxr  # pylint: disable=unused-import  # noqa: F401

from geokube import (
    Collection,
    CoordinateSystem,
    CRS,
    Cube,
    Geodetic,
    RotatedGeodetic,
    axis,
)
from geokube.core import domain, field
from geokube.core.cell_method import CellMethod


_FEATURE_MAP: dict[str | None, type[field.Field]] = {
    "point": field.PointsField,
    # 'timeSeries': ...,
    # 'trajectory': ...,
    "profile": field.ProfilesField,
    # 'timeSeriesProfile': ...,
    # 'trajectoryProfile': ...,
    None: field.GridField,
}


def _create_field(dset: xr.Dataset, variable: str) -> field.Field:
    dset_cf = dset.cf
    dset_coords = dict(dset.coords)

    # Horizontal coordinate system. -------------------------------------------
    if gmn := dset_cf.grid_mapping_names:
        crs_var_name = next(iter(gmn.values()))[0]
        hor_crs = CRS.from_cf(dset[crs_var_name].attrs)
        dset_coords.pop(crs_var_name)
    else:
        hor_crs = Geodetic()

    # Scalars to coordinates. -------------------------------------------------
    for dim, coord in dset_coords.items():
        if not coord.dims:
            dset = dset.expand_dims(dim=[dim], axis=0)
            dset_coords[dim] = dset[dim]
            dset_cf = dset.cf

    # Axis names. -------------------------------------------------------------
    axes = {}
    for standard_name, name in dset_cf.standard_names.items():
        assert len(name) == 1
        if (axis_ := axis._from_string(standard_name)) is not None:
            axes[name[0]] = axis_
    for dim in dset.dims:
        if dim not in axes and (axis_ := axis._from_string(dim)) is not None:
            axes[dim] = axis_

    # Coordinates. ------------------------------------------------------------
    coords = {}
    for cf_coord, cf_coord_names in dset_cf.coordinates.items():
        assert len(cf_coord_names) == 1
        cf_coord_name = cf_coord_names[0]
        coord = dset_coords.pop(cf_coord_name)
        coord.attrs.setdefault("units", "dimensionless")
        axis_ = axis._from_string(cf_coord)
        coords[axis_] = xr.Variable(
            dims=tuple(axes[dim] for dim in coord.dims),
            data=coord.to_numpy(),
            attrs=coord.attrs,
            encoding=coord.encoding,
        )
    for cf_axis, cf_axis_names in dset_cf.axes.items():
        assert len(cf_axis_names) == 1
        cf_axis_name = cf_axis_names[0]
        if cf_axis_name in dset_coords:
            coord = dset_coords.pop(cf_axis_name)
            coord.attrs.setdefault("units", "dimensionless")
            axis_ = axis._from_string(cf_axis.lower())
            if isinstance(hor_crs, RotatedGeodetic):
                if axis_ is axis.x:
                    axis_ = axis.grid_longitude
                elif axis_ is axis.y:
                    axis_ = axis.grid_latitude
            coords[axis_] = xr.Variable(
                dims=tuple(axes[dim] for dim in coord.dims),
                data=coord.to_numpy(),
                attrs=coord.attrs,
                encoding=coord.encoding,
            )
    # Time bounds.
    if (cm_attr := dset[variable].attrs.get("cell_methods")) is not None:
        cell_method = CellMethod.parse(cm_attr)
        cm_axis = cell_method.axis
        cm_time = cm_axis == "time" or "time" in cm_axis
    else:
        cm_time = False
    if cm_time and (
        t_bnds := (dset.cf.bounds.get("time") or dset.cf.bounds.get("T"))
    ):
        assert len(t_bnds) == 1
        time_vals = dset[t_bnds[0]].to_numpy()
        time_bnds = pd.IntervalIndex.from_arrays(
            left=time_vals[:, 0], right=time_vals[:, 1], closed="both"
        )
        time_coord = coords[axis.time]
        coords[axis.time] = xr.Variable(
            dims=time_coord.dims,
            data=time_bnds,
            attrs=time_coord.attrs,
            encoding=time_coord.encoding,
        )

    # Coordinate system. ------------------------------------------------------
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
        time=time.pop() if time else None,
    )

    # Domain. -----------------------------------------------------------------
    result_dset = domain.Domain.as_xarray_dataset(coords, crs)
    result_dset = result_dset.drop_indexes(result_dset.xindexes)
    result_dset.encoding.update(dset.encoding)

    # Data. -------------------------------------------------------------------
    var = dset[variable]
    data = xr.Variable(
        dims=tuple(axes[dim] for dim in var.dims),
        data=var.data,
        attrs=var.attrs,
        encoding=var.encoding,
    )
    data.attrs["grid_mapping"] = result_dset.attrs["grid_mapping"]
    data.attrs.setdefault("units", "dimensionless")
    data_vars = {variable: data}
    result_dset = result_dset.assign(data_vars)
    result_dset.attrs.update(dset.attrs)

    # Field. ------------------------------------------------------------------
    field_type = _FEATURE_MAP[dset.attrs.get("featureType")]
    result = field_type._from_xarray_dataset(result_dset)
    return result


def _create_all_fields(
    dset: xr.Dataset, variables: list[str]
) -> tuple[field.Field, ...]:
    fields = []
    for var in variables:
        darr = dset[var]
        darr_cf = darr.cf

        # Finding redundant standard names.
        std_names = set.union(
            *[set(names) for names in darr_cf.standard_names.values()]
        )
        axes = set.union(*[set(names) for names in darr_cf.axes.values()])
        coords = set.union(
            *[set(names) for names in darr_cf.coordinates.values()]
        )
        redundant_names = std_names - (axes | coords)

        # Creating a new dataset, with a single variable.
        new_dset = darr.to_dataset().drop_vars(names=redundant_names)
        new_dset.attrs |= dset.attrs
        new_dset.encoding |= dset.encoding

        # Fixing bounds.
        for coord in new_dset.coords.values():
            if (bound_name := coord.encoding.get("bounds")) is not None:
                bounds = {bound_name: dset[bound_name].variable}
                new_dset = new_dset.assign_coords(coords=bounds)

        # Creating and adding the corresponding field.
        new_field = _create_field(new_dset, var)
        fields.append(new_field)

    return tuple(fields)


def _group_fields(fields: list[field.Field]) -> Collection:
    start = fields[0]
    groups: list[list[field.Field]] = [[start]]
    domains = [start.domain]
    for field_ in fields[1:]:
        current_domain = field_.domain
        for group_domain, group in zip(domains, groups):
            if current_domain == group_domain:
                group.append(field_)
                break
        else:
            groups.append([field_])
            domains.append(current_domain)
    cubes = [Cube(fields=group) for group in groups]
    return Collection(data=cubes)


def open(
    path: str | Sequence[str | os.PathLike],
    variables: str | Sequence[str] | None = None,
    xarray_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> Collection:
    """
    Return a collection of fields from file(s).

    Parameters
    ----------
    path : str, os.PathLike, or sequence
        Path to the file(s).
    variables : str or sequence of str, default: None
        Variables to extract from the file(s).
    xarray_kwargs : dict | None, optional
        Additional keyword arguments passed to the function
        xarray.open_mfdataset.
    **kwargs : dict, optional
        Additional keyword arguments.  Used for consistency among the
        drivers.  Ignored.

    Returns
    -------
    Collection
        Collection of fields from file(s).

    Raises
    ------
    TypeError
        If `variables` is not str, sequence of str, or None.

    """
    kwa = xarray_kwargs or {}
    kwa.setdefault("decode_coords", "all")
    # TODO: Reconsider using the context manager.
    # TODO: Consider the attributes and encoding of the return collection.
    with xr.open_mfdataset(path, **kwa) as dset:
        match variables:
            case str():
                var_names = [variables]
            case Sequence():
                var_names = list(variables)
            case None:
                var_names = list(dset.data_vars.keys())
            case _:
                raise TypeError(
                    "'variables' can be a 'str', sequence of 'str', or "
                    f"'None'; {type(variables)} is not supported"
                )
        fields = _create_all_fields(dset, var_names)
        coll = _group_fields(fields)
        return coll
