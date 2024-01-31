import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint


def create_quantity(
    values: npt.ArrayLike | pint.Quantity,
    default_units: pint.Unit | None,
    default_dtype: np.dtype,
) -> pint.Quantity:
    match values:
        # case pint.Quantity() if isinstance(values.magnitude, np.ndarray):
        #     return values
        # case pint.Quantity():
        #     return pint.Quantity(np.asarray(values.magnitude), values.units)
        case pint.Quantity():
            return (
                values
                if isinstance(values.magnitude, np.ndarray)
                else pint.Quantity(np.asarray(values.magnitude), values.units)
            )
        case np.ndarray():
            # NOTE: The pattern arr * unit does not work when arr has stings.
            return pint.Quantity(values, default_units)
        case da.Array():
            return pint.Quantity(values.compute(), default_units)
        case pd.IntervalIndex():
            return pint.Quantity(
                np.asarray(values, dtype=object), default_units
            )
        case _:
            return pint.Quantity(
                np.asarray(values, dtype=default_dtype), default_units
            )


def is_monotonic(values: pint.Quantity) -> bool:
    mag = values.magnitude
    mag_start, mag_stop = mag[:-1], mag[1:]
    return bool(np.all(mag_start < mag_stop) or np.all(mag_start > mag_stop))


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
