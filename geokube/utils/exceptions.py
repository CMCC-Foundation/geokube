# -- Exceptions ---------------------------------------------------------------
from ..utils.hcube_logger import HCubeLogger


class NonUniqueKey(KeyError):
    """Provided key is not unique"""


class NotSupportedError(RuntimeError):
    """The operation is not supported"""


class HCubeNoSuchAxisError(KeyError):
    """Requested axis does not exist"""

    def __init__(self, axis_name: str, logger: HCubeLogger):
        msg = f"Axis `{axis_name}` does not exist!"
        super().__init__(msg)
        logger.error(msg)


class HCubeTypeError(TypeError):
    """geokube wrapper around TypeError"""

    def __init__(self, msg, logger: HCubeLogger):
        super().__init__(msg)
        logger.error(msg)


class HCubeValueError(ValueError):
    """geokube wrapper around ValueError"""

    def __init__(self, msg, logger: HCubeLogger):
        super().__init__(msg)
        logger.error(msg)


class HCubeKeyError(KeyError):
    """geokube wrapper around KeyError"""

    def __init__(self, msg, logger: HCubeLogger):
        super().__init__(msg)
        logger.error(msg)


class HCubeNotImplementedError(NotImplementedError):
    """geokube wrapper around NotImplementedError"""

    def __init__(self, msg, logger: HCubeLogger):
        super().__init__(msg)
        logger.error(msg)
