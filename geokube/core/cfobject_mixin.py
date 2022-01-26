import geokube.utils.exceptions as ex


class CFObjectMixin:
    def apply_from_other(self, other):
        if not isinstance(other, type(self)):
            raise ex.HCubeTypeError(
                f"Type of `other` argument must comply with `{type(self)}` type!"
            )
        for att_name in self.__slots__:
            setattr(self, att_name, getattr(other, att_name))
