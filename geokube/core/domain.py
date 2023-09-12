from typing import Mapping, Sequence

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis
from .coord_system import CoordinateSystem


class Points:
    __slots__ = ('__coords', '__coord_system', '__n_pts')

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coord_system, CoordinateSystem):
            raise TypeError(
                "'coord_system' must be an instance of 'CoordinateSystem'"
            )
        self.__coord_system = coord_system

        units = coord_system.units
        pts = ('_points',)

        if isinstance(coords, Mapping):
            result_coords = {}
            n_pts = set()
            for axis_, vals in coords.items():
                match vals:
                    case pint.Quantity():
                        vals_ = vals
                    case np.ndarray():
                        # NOTE: The pattern arr * unit does not work when arr
                        # has stings.
                        vals_ = pint.Quantity(vals, units.get(axis_))
                    case da.Array():
                        vals_ = pint.Quantity(vals.compute(), units.get(axis_))
                    case _:
                        vals_ = pint.Quantity(
                            np.asarray(vals, dtype=axis_.dtype),
                            units.get(axis_)
                        )
                if vals_.ndim != 1:
                    raise ValueError(
                        "'coords' must have only one-dimensional values"
                    )
                result_coords[axis_] = xr.DataArray(vals_, dims=pts)
                n_pts.add(vals_.size)
            if len(n_pts) != 1:
                raise ValueError("'coords' must have values of equal sizes")
            self.__n_pts = n_pts.pop()
            self.__coords = result_coords
        elif isinstance(coords, Sequence):
            # NOTE: This approach currently does not allows providing units.
            n_dims = {len(point) for point in coords}
            if len(n_dims) != 1:
                raise ValueError(
                    "'coords' must have points of equal number of dimensions"
                )
            self.__n_pts = len(coords)
            data = pd.DataFrame(data=coords, columns=coord_system.axes)
            self.__coords = {
                axis_: xr.DataArray(
                    pint.Quantity(
                        vals.to_numpy(dtype=axis_.dtype), units.get(axis_)
                    ),
                    dims=pts
                )
                for axis_, vals in data.items()
            }
        else:
            raise TypeError("'coords' must be a sequence or mapping")

    @property
    def coord_system(self) -> CoordinateSystem:
        return self.__coord_system

    @property
    def _coords(self) -> dict[axis.Axis, xr.DataArray]:
        return self.__coords

    @property
    def _n_pts(self) -> int:
        return self.__n_pts
