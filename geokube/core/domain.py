from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis, coord_system


class Points:
    __slots__ = ('__coords', '__coord_syst', '__n_pts')

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_syst: coord_system.CoordinateSystem
    ) -> None:
        if not isinstance(coord_syst, coord_system.CoordinateSystem):
            raise TypeError(
                "'coord_syst' must be an instance of 'CoordinateSystem'"
            )
        self.__coord_syst = coord_syst

        units = coord_syst.units
        pts = ('_points',)

        if isinstance(coords, Mapping):
            result_coords = {}
            n_pts = set()
            for axis_, vals in coords.items():
                match vals:
                    case pint.Quantity():
                        vals_ = vals
                    case _:
                        # NOTE: The pattern arr * unit does not work when arr
                        # has stings.
                        arr = np.asarray(vals, dtype=_DTYPES.get(axis_))
                        vals_ = pint.Quantity(arr, units.get(axis_))
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
            data = pd.DataFrame(data=coords, columns=coord_syst.all_axes)
            self.__coords = {
                axis_: xr.DataArray(
                    pint.Quantity(
                        vals.to_numpy(dtype=_DTYPES.get(axis_)),
                        units.get(axis_)
                    ),
                    dims=pts
                )
                for axis_, vals in data.items()
            }
        else:
            raise TypeError("'coords' must be a sequence or mapping")

    @property
    def coord_syst(self) -> coord_system.CoordinateSystem:
        return self.__coord_syst

    @property
    def _coords(self) -> dict[axis.Axis, xr.DataArray]:
        return self.__coords

    @property
    def _n_pts(self) -> int:
        return self.__n_pts


_DTYPES = {axis.time: 'datetime64', axis.timedelta: 'timedelta64'}
