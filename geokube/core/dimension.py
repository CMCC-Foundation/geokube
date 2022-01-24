from .axis import Axis, AxisType
from geokube.utils.hcube_logger import HCubeLogger
from typing import Mapping, Any, Union

class Dimension(Axis):

    __slots__ = ("_encoding")

    _LOG = HCubeLogger(name="Dimension")

    def __init__(
        self,
        name: Union[Axis,str],
        axistype: AxisType=None,
        encoding: Mapping[Any, Any]=None,
    ):
        super().__init__(name, axistype)
        self._encoding = encoding

    @property
    def ncvar(self):
        return self._encoding.get("name", self.name)