from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Self

import numpy as np
import xarray as xr


# TODO: Consider making this module and `TwoDimIndex` internal.


@dataclass(frozen=True, slots=True)
class TwoDimIndex(xr.core.indexes.Index):
    variables: dict[str, xr.Variable]
    dims: tuple[Hashable, ...]

    # TODO: Consider using `Axis` instead of `str` for dims.

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> Self:
        variables_ = dict(variables)

        if len(variables_) != 2:
            raise ValueError("'variables' must be a mapping with two items")

        vars_ = iter(variables.values())
        dims = next(vars_).dims
        if next(vars_).dims != dims:
            raise ValueError("'variables' must have same dimensions")
        if len(dims) != 2:
            raise ValueError("'variables' must be two-dimensional")

        return cls(variables_, dims)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | xr.Variable]
    ) -> Self | None:
        # TODO: Consider if this method is going to be implemented.
        pass

    def sel(self, labels: dict[Any, Any]) -> xr.core.indexing.IndexSelResult:
        dims = self.dims
        vars_ = self.variables
        vars_iter = iter(vars_)
        x_name = next(vars_iter)
        x_vals = vars_[x_name].to_numpy()
        y_name = next(vars_iter)
        y_vals = vars_[y_name].to_numpy()

        indices = {}

        for name, var in vars_.items():
            if (idx := labels.get(name)) is not None:
                if isinstance(idx, slice):
                    # TODO: Consider the case when `idx` has `None` for `start`
                    # or `stop`.
                    slices = True
                    lower_bound, upper_bound = sorted([idx.start, idx.stop])
                    vals = var.to_numpy()
                    indices[name] = (
                        (vals >= lower_bound) & (vals <= upper_bound)
                    )
                else:
                    indices[name] = np.array(idx, ndmin=1)
                    slices = False

        x_idx = indices[x_name]
        y_idx = indices[y_name]
        if slices:
            nonzero_idx = np.nonzero(x_idx & y_idx)
            sel_idx = {
                dim: slice(incl_idx.min(), incl_idx.max() + 1)
                for dim, incl_idx in zip(self.dims, nonzero_idx)
            }
        else:
            # Calculating the squares of the Euclidean distance.
            x_data = x_vals[np.newaxis, :, :]
            y_data = y_vals[np.newaxis, :, :]
            x_pts = x_idx[:, np.newaxis, np.newaxis]
            y_pts = y_idx[:, np.newaxis, np.newaxis]
            # x_pts = x_idx.reshape(-1, 1, 1)
            # y_pts = y_idx.reshape(-1, 1, 1)
            x_diff = x_data - x_pts
            y_diff = y_data - y_pts
            sum_sq_diff = x_diff * x_diff + y_diff * y_diff

            # Selecting the indices that correspond to the squares of the
            # Euclidean distance.
            # TODO: Improve vectorization.
            # TODO: Consider replacing `numpy.unravel_index` with
            # `numpy.argwhere`, using the constructs like:
            # `np.argwhere(diff_sq[i] == diff_sq[i].min())[0]`.
            n_dims, *shape = sum_sq_diff.shape
            idx_nd = tuple(
                np.unravel_index(indices=sum_sq_diff[i].argmin(), shape=shape)
                for i in range(n_dims)
            )
            idx_ = np.array(idx_nd, dtype=np.int64)
            sel_idx = {dims[0]: idx_[:, 0], dims[1]: idx_[:, 1]}

        return xr.core.indexing.IndexSelResult(sel_idx)
