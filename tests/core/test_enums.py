import pytest

from geokube.core.enums import MethodType


class TestEnums:
    def test_max_methodtype(self):
        mt = MethodType("max")
        assert mt is MethodType.MAX
