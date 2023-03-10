from .variable import Variable


class Bounds(Variable):
    __slots__ = ("_name",)

    @property
    def name(self):
        return self._name


class Bounds1D(Bounds):
    pass


class BoundsND(Bounds):
    pass
