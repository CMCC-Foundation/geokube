from dataclasses import dataclass, asdict
from typing import SupportsFloat

import numpy as np
import numpy.typing as npt
import pint

from . import axis


@dataclass(init=False, frozen=True, slots=True)
class BBox:
    south: SupportsFloat | pint.Quantity | None
    north: SupportsFloat | pint.Quantity | None
    west: SupportsFloat | pint.Quantity | None
    east: SupportsFloat | pint.Quantity | None
    bottom: SupportsFloat | pint.Quantity | None
    top: SupportsFloat | pint.Quantity | None

    def __init__(self, *values: axis.AxisIndexer) -> None:
        ax_idx: dict[axis.Axis, slice | npt.ArrayLike | pint.Quantity]
        ax_idx = dict(values)

        if (lat := ax_idx.pop(axis.latitude, None)) is None:
            south, north = None, None
        elif isinstance(lat, slice):
            bounds = lat.start, lat.stop
            south, north = bounds if None in bounds else sorted((bounds))
        else:
            raise TypeError("'lat' must be slice")
        if (lon := ax_idx.pop(axis.longitude, None)) is None:
            west, east = None, None
        elif isinstance(lon, slice):
            bounds = lon.start, lon.stop
            west, east = bounds if None in bounds else sorted((bounds))
        else:
            raise TypeError("'lon' must be slice")
        if (vert := ax_idx.pop(axis.vertical, None)) is None:
            bottom, top = None, None
        elif isinstance(vert, slice):
            bounds = vert.start, vert.stop
            bottom, top = bounds if None in bounds else sorted((bounds))
        else:
            raise TypeError("'vert' must be slice")

        if ax_idx:
            raise ValueError(
                "'values'' contain axes that are not allowed: "
                f"{sorted(ax_idx.keys())}"
            )

        cls_ = type(self)
        cls_.south.__set__(self, south)
        cls_.north.__set__(self, north)
        cls_.west.__set__(self, west)
        cls_.east.__set__(self, east)
        cls_.bottom.__set__(self, bottom)
        cls_.top.__set__(self, top)

    def to_dict(self) -> dict[str, SupportsFloat | pint.Quantity | None]:
        # NOTE: The function `dataclasses.asdict` makes deep copies. In this
        # case only scalars are expected, so it sould not be an issue.
        return asdict(self)
        # return {
        #     'north': self.north,
        #     'west': self.west,
        #     'south': self.south,
        #     'east': self.east,
        #     'top': self.top,
        #     'bottom': self.bottom
        # }


class Shape:
    # TODO: Implement this.
    pass


class Nuts:
    # TODO: Implement this.
    pass
