class AggMixin:
    def min(self):
        # TODO: maybe check if attribute defined
        return self._variable.min()

    def max(self):
        # TODO: maybe check if attribute defined
        return self._variable.max()

    def first(self):
        return self._variable[0]

    def first(self):
        return self._variable[-1]
