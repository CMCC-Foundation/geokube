import json

import numpy as np
import dask.array as da


def _as_concrete_datetime64(obj):
    """Coerce a generic-unit ``datetime64`` (scalar or array) to ``datetime64[ns]``.

    ``str()`` on -- and ``np.nanmin`` over -- a generic-unit ``datetime64`` raises ("Cannot
    convert a NumPy datetime value other than NaT with generic units"); a concrete resolution is
    always serializable. Non-generic ``datetime64`` values pass through unchanged."""
    if np.datetime_data(obj.dtype)[0] == "generic":
        return obj.astype("datetime64[ns]")
    return obj


class GeokubeDetailsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.datetime64):
            return str(_as_concrete_datetime64(obj))
        return json.JSONEncoder.default(self, obj)


def maybe_convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.float32) or np.issubdtype(
            obj.dtype, np.float64
        ):
            return obj.astype(float).tolist()
        elif np.issubdtype(obj.dtype, np.int32) or np.issubdtype(
            obj.dtype, np.int64
        ):
            return obj.astype(int).tolist()
        elif np.issubdtype(obj.dtype, np.datetime64):
            return _as_concrete_datetime64(obj).astype(str).tolist()
        else:
            return obj.tolist()
    elif isinstance(obj, da.Array):
        return maybe_convert_to_json_serializable(np.array(obj))
    elif isinstance(obj, dict):
        return {
            k: maybe_convert_to_json_serializable(v) for k, v in obj.items()
        }
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.datetime64):
        return str(_as_concrete_datetime64(obj))
    else:
        return obj
