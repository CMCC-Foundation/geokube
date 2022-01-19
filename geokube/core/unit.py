import cf_units as cf


class Unit:

    __slots__ = (
        "_unit",
        "_backup_name",
    )

    def __init__(self, unit, calendar=None):
        try:
            self._unit = cf.Unit(unit=unit, calendar=calendar)
            self._backup_name = None
        except ValueError:
            self._unit = cf.Unit(unit=None, calendar=calendar)
            self._backup_name = unit

    @property
    def is_unknown(self):
        return self._backup_name is not None

    def __str__(self):
        if self._backup_name is None:
            return str(self._unit)
        return self._backup_name

    def __getattr__(self, name):
        if name not in vars(self._unit):
            raise AttributeError(f"Attribute `{name}` is not available.")
        return getattr(self._unit, name)

    def __getstate__(self):
        return dict(**self._unit.__getstate__(), **{"backup_name": self._backup_name})

    def __setstate__(self, state):
        self.__init__(state["unit_text"], calendar=state["calendar"])
