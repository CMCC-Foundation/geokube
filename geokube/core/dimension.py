from __future__ import annotations
from .axis import Axis, AxisType
from geokube.utils.hcube_logger import HCubeLogger
from typing import Mapping, Any, Union

class Dimension(Axis):

    __slots__ = ("_encoding")

    _LOG = HCubeLogger(name="Dimension")

    def __init__(
        self,
        name: Union[str, Axis, Dimension],
        axistype: AxisType=None,
        encoding: Mapping[Any, Any]=None,
    ):
        if isinstance(name, Dimension):
            self.copy(name)
        else:
            super().__init__(name, axistype)
            self._encoding = encoding

    @property
    def ncvar(self):
        return self._encoding.get("name", self.name)
    
    @property
    def encoding(self):
        return self._encoding

    def copy(self, other):
        self._name = other.name
        self._type = other.type
        self._encoding = other.encoding

    def __repr__(self) -> str:
        return super().__repr__()