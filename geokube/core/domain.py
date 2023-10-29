from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import pint_xarray
import xarray as xr

from . import axis
from .coord_system import CoordinateSystem
from .feature import GridFeature, PointsFeature, ProfilesFeature
from .quantity import create_quantity


# TODO: Consider renaming this class to `DomainMixin`.
class Domain:
    __slots__ = ()

    @classmethod
    def _from_xrdset(
        cls, dset: xr.Dataset, coord_system: CoordinateSystem
    ) -> Self:
        return cls(coords=dset.coords, coord_system=coord_system)


class Points(Domain, PointsFeature):
    __slots__ = ()

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_system: CoordinateSystem
    ) -> None:
        units = coord_system.units

        match coords:
            case Mapping():
                result_coords = {}
                for axis_, coord in coords.items():
                    result_coords[axis_] = qty = create_quantity(
                        coord, units.get(axis_), axis_.encoding['dtype']
                    )
                    if qty.ndim != 1:
                        raise ValueError(
                            f"'coords' have axis {axis_} that does not have "
                            "one-dimensional values"
                        )
                if not set(coord_system.axes) <= result_coords.keys():
                    raise ValueError(
                        "'coords' must have all axes contained in the "
                        "coordinate system"
                    )
            case Sequence():
                # NOTE: This approach currently does not allow providing units.
                n_dims = {len(point) for point in coords}
                if len(n_dims) != 1:
                    raise ValueError(
                        "'coords' must have points of equal number of "
                        "dimensions"
                    )
                data = pd.DataFrame(data=coords, columns=coord_system.axes)
                result_coords = {
                    axis_: pint.Quantity(
                        vals.to_numpy(dtype=axis_.encoding['dtype']),
                        units.get(axis_)
                    )
                    for axis_, vals in data.items()
                }
            case _:
                raise TypeError("'coords' must be a sequence or mapping")

        super().__init__(coords=result_coords, coord_system=coord_system)


class Profiles(Domain, ProfilesFeature):

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        units = coord_system.units
        interm_coords = dict(coords)
        result_coords: dict[axis.Axis, xr.DataArray] = {}
        prof = (self._DIMS_[0],)
        n_prof = set()
        vert = interm_coords.pop(axis.vertical)

        # Vertical.
        if isinstance(vert, Sequence):
            n_lev = [len(vert_val) for vert_val in vert]
            n_lev_tot = max(n_lev)
            n_prof_tot = len(vert)
            vert_vals = np.empty(
                shape=(n_prof_tot, n_lev_tot), dtype=axis.vertical.dtype
            )
            for i, (stop_idx, vals) in enumerate(zip(n_lev, vert)):
                if stop_idx == n_lev_tot:
                    vert_vals[i, :] = vals
                else:
                    vert_vals[i, :stop_idx] = vals
                    vert_vals[i, stop_idx:] = np.nan
            vert_ = pint.Quantity(vert_vals, units[axis.vertical])
        else:
            vert_ = create_quantity(
                vert, units.get(axis.vertical), axis.vertical.dtype
            )
            vert_shape = vert_.shape
            if len(vert_shape) != 2:
                raise ValueError(
                    "'coords' must have vertical as a two-dimensional data "
                    "structure"
                )
            n_prof_tot, n_lev_tot = vert_shape

        result_coords[axis.vertical] = xr.DataArray(
            vert_, dims=self._DIMS_
        )

        # All coordinates except the vertical.
        for axis_, vals in interm_coords.items():
            vals_ = create_quantity(vals, units.get(axis_), axis_.dtype)
            if vals_.ndim != 1:
                raise ValueError(
                    f"'coords' have axis {axis_} that does not have "
                    "one-dimensional values"
                )
            result_coords[axis_] = xr.DataArray(vals_, dims=prof)
            n_prof.add(vals_.size)
        if len(n_prof) != 1:
            raise ValueError(
                "'coords' with the exception of vertical must have values of "
                "equal sizes"
            )
        if n_prof_tot != n_prof.pop():
            raise ValueError("'coords' have items of with inappropriate sizes")
        if not set(coord_system.axes) <= result_coords.keys():
            raise ValueError(
                "'coords' must have all axes from the coordinate system"
            )
        self._n_profiles = n_prof_tot
        self._n_levels = n_lev_tot

        super.__init__(coords = result_coords,
                       coord_system=coord_system)

class Grid(Domain, GridFeature):
    # TODO: Consider auxiliary coordinates other than the
    # latitude and longitude. Especially consider how to represent them in the
    # API.
    # NOTE: The assumption is that the latitude and longitude as auxiliary
    # coordinates must have the dimensions either
    # `(axis.grid_latitude, axis.grid_longitude)` or `(axis.y, axis.x)`.
    __slots__ = ()

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        crs = coord_system.spatial.crs
        self._DIMS_ = crs.dim_axes

        units = coord_system.units
        result_coords: dict[axis.Axis, xr.DataArray] = {}

        hor_aux_shapes = set()

        for axis_ in crs.dim_axes: # TODO: REVIEW! - we need to keep the order as in the CRS
            vals_ = create_quantity(coords[axis_], units.get(axis_), axis_.dtype)
            if axis_ in coord_system.dim_axes:
                # Dimension coordinates.
                if not vals_.ndim:
                    dim_axes = ()
                elif vals_.ndim == 1:
                    dim_axes = (axis_,)
                else:
                    raise ValueError(
                        f"'coords' have a dimension axis {axis_} that has "
                        "multi-dimensional values"
                    )
            else:
                # Auxiliary coordinates.
                dim_axes = crs.dim_axes if vals_.ndim else ()
                if axis_ in crs.aux_axes:
                    hor_aux_shapes.add(vals_.shape)
            
            #
            # dequantify is needed because pandas index do not keep quantity 
            # in the coordinates
            # -> dequantify put units as attributes in the dataset
            # we need to add also cf-attributes
            # 
            result_coords[axis_] = xr.DataArray(vals_, 
                                                dims=dim_axes,
                                                attrs=axis_.encoding).pint.dequantify()
        if len(hor_aux_shapes) > 1:
            raise ValueError(
                "'coords' have auxiliary horizontal coordinates with different"
                "shapes"
            )

        super().__init__(result_coords, coord_system)
