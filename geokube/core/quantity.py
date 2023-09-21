import dask.array as da
import numpy as np
import numpy.typing as npt
import pint


def _get_array_like_magnitude(
    data: npt.ArrayLike | da.Array | pint.Quantity, units: pint.Unit
) -> npt.ArrayLike | da.Array:
    if isinstance(data, pint.Quantity):
        if data.units != units:
            data = data.to(units)
        return data.magnitude
    return data


def _get_slice_magnitude(data: slice, units: pint.Unit) -> slice:
    if data.step is not None:
        raise ValueError("'data' must have the step 'None'")
    start = _get_array_like_magnitude(data.start, units)
    stop = _get_array_like_magnitude(data.stop, units)
    return slice(start, stop)


def get_magnitude(
    data: slice | npt.ArrayLike | da.Array | pint.Quantity, units: pint.Unit
) -> slice | npt.ArrayLike | da.Array:
    match data:
        case slice():
            return _get_slice_magnitude(data, units)
        case _:
            return _get_array_like_magnitude(data, units)
