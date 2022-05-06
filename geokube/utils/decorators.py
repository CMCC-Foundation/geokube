import warnings
from functools import wraps


def nonclose_assert(f):
    @wraps(f)
    def decorated_function(self, *args, **kwargs):
        if hasattr(self, "_is_closed") and self._is_closed:
            raise RuntimeError(
                "This object is already closed. Reinitialize it to do "
                "computations!"
            )
        return f(self, *args, **kwargs)

    return decorated_function


def geokube_logging(f):
    @wraps(f)
    def decorated_function(self, *args, **kwargs):
        if not hasattr(self, "_LOG"):
            warnings.warn(
                f"The class of the decorated `{str(f)}` does not contain"
                " `_LOG` object."
            )
            return f(self, *args, **kwargs)
        self._LOG.debug(f"Entering `{str(f)}`")
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            self._LOG.error(
                f"{type(e).__name__} with message `{e}` raised while executing"
                f" `{str(f)}`"
            )
            raise e
        finally:
            self._LOG.debug(f"Exiting `{str(f)}`")

    return decorated_function
