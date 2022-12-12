import json

import numpy as np
import dask.array as da


class GeokubeDetailsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.datetime64):
            return str(obj)
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
            return obj.astype(str).tolist()
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
        return str(obj)
    else:
        return obj
