import json

import numpy as np


class GeokubeDetailsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.datetime64):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def maybe_convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.float32):
            return list(obj.astype(float))
        elif np.issubdtype(obj.dtype, np.datetime64):
            return list(obj.astype(str))
        else:
            return list(obj)
    elif isinstance(obj, dict):
        return {
            k: maybe_convert_to_json_serializable(v) for k, v in obj.items()
        }
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.datetime64):
        return str(obj)
    else:
        return obj
