import logging


class HCubeLogger:
    __slots__ = ("_logger",)

    def __init__(self, name):
        self._logger = logging.getLogger(name)

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def warn(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)
