from geokube.utils.hcube_logger import HCubeLogger
import geokube.utils.exceptions as ex
import abc


class CFObjectAbstract(metaclass=abc.ABCMeta):

    _LOG = HCubeLogger(name="CFObjectAbstract")

    @abc.abstractmethod
    def apply_from_other(self, other, shallow=False):
        raise NotImplementedError
