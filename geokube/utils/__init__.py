from geokube.utils.exceptions import NonUniqueKey


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise NonUniqueKey(
                f"Provided key `{key}` is already in stored! Keys should be unique!"
            )
        super().__setitem__(key, value)
