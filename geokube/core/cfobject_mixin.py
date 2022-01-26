import geokube.utils.exceptions as ex


class CFObjectMixin:
    def apply_from_other(self, other, shallow=False):
        if not issubclass(type(other), type(self)):
            raise ex.HCubeTypeError(
                f"`other` argument must be of the type `{type(self)}` or its subclass!"
            )
        if shallow:
            for att_name in self.__slots__:
                setattr(self, att_name, getattr(other, att_name))
        else:
            for supertype in type(self).__mro__:
                if not hasattr(supertype, "__slots__"):
                    continue
                for att_name in supertype.__slots__:
                    setattr(self, att_name, getattr(other, att_name))
