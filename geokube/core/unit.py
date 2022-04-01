import cf_units as cf


class Unit:
    UNKNOWN_CF_UNIT = cf.Unit(None)

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
        return (
            self._backup_name is not None or self._unit == Unit.UNKNOWN_CF_UNIT
        )

    def __repr__(self):
        return f"<Unit({str(self._unit)}, backeup_name={self._backup_name}>"

    def __str__(self):
        if self._backup_name is None:
            return str(self._unit)
        return self._backup_name

    def __getattr__(self, name):
        if not hasattr(self._unit, name):
            raise AttributeError(f"Attribute `{name}` is not available.")
        return getattr(self._unit, name)

    def __getstate__(self):
        return dict(
            **self._unit.__getstate__(), **{"backup_name": self._backup_name}
        )

    def __setstate__(self, state):
        self.__init__(state["unit_text"], calendar=state["calendar"])

    def __eq__(self, other):
        if not isinstance(other, Unit):
            return False
        return (
            self._unit == other._unit
            and self._backup_name == other._backup_name
        )

    def __ne__(self, other):
        return not (self == other)
