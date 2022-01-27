from geokube.utils.hcube_logger import HCubeLogger
import geokube.utils.exceptions as ex


class CFObjectMixin:

    _LOG = HCubeLogger(name="CFObjectMixin")

    def apply_from_other(self, other, shallow=False):
        if shallow:
            if not hasattr(other, "__slots__") or not hasattr(self, "__slots__"):
                return
            for att_name in other.__slots__:
                try:
                    setattr(self, att_name, getattr(other, att_name))
                except:
                    pass
        else:
            for supertype in type(other).__mro__:
                if not hasattr(supertype, "__slots__") or not hasattr(
                    self, "__slots__"
                ):
                    continue
                for att_name in supertype.__slots__:
                    try:
                        setattr(self, att_name, getattr(other, att_name))
                    except:
                        pass
