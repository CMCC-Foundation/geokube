from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Callable, Self, TYPE_CHECKING

import dask.array as da
import numpy as np


class _Method(StrEnum):
    # NOTE: See
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ape.html
    POINT = auto()
    SUM = auto()
    MAXIMUM = auto()
    MEDIAN = auto()
    # MID_RANGE = auto()
    MINIMUM = auto()
    MEAN = auto()
    # MODE = auto()
    # RANGE = auto()
    STANDARD_DEVIATION = auto()
    VARIANCE = auto()
    UNDEFINED = ''

    @classmethod
    def _missing_(cls, value):
        val = value.lower()
        for member in cls:
            if member.value == val:
                return member
        return cls.UNDEFINED


@dataclass(frozen=True, slots=True)
class _MethodInfo:
    numpy_func: Callable
    dask_func: Callable


class CellMethod:
    # NOTE: See
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch07s03.html
    # NOTE: Currently supports non-combined cell methods.
    # TODO: Extend for combined ones and special variables, like
    # `lat: lon: standard_deviation`, `area: mean`, etc.

    __slots__ = ('_method', '_axis', '_interval', '_comment', '_where')

    if TYPE_CHECKING:
        _method: _Method
        _axis: str | tuple[str, ...] | None
        _interval: str
        _comment: str
        _where: str

    _METHOD_MAP: dict[_Method, _MethodInfo | None] = {
        _Method.POINT: None,
        _Method.SUM: _MethodInfo(np.nansum, da.nansum),
        _Method.MAXIMUM: _MethodInfo(np.nanmax, da.nanmax),
        _Method.MEDIAN: _MethodInfo(np.nanmedian, da.nanmedian),
        # TODO: Implement `mid_range`.
        # _Method.MID_RANGE: _MethodInfo(..., ...),
        _Method.MINIMUM: _MethodInfo(np.nanmin, da.nanmin),
        _Method.MEAN: _MethodInfo(np.nanmean, da.nanmean),
        # TODO: Implement `mode`.
        # _Method.MODE = _MethodInfo(..., ...),
        # TODO: Implement `range` with `nan*` behavior.
        # _Method.RANGE = _MethodInfo(np.ptp, da.ptp),
        _Method.STANDARD_DEVIATION: _MethodInfo(np.nanstd, da.nanstd),
        _Method.VARIANCE: _MethodInfo(np.nanvar, da.nanvar),
        _Method.UNDEFINED: None
    }

    def __init__(
        self,
        method: str = '',
        axis: str | Sequence[str, ...] | None = None,
        interval: str = '',
        comment: str = '',
        where: str = '',
    ) -> None:
        self._method = _Method(method)
        self._axis = axis if isinstance(axis, str) else tuple(axis)
        self._interval = str(interval)
        self._comment = str(comment)
        self._where = str(where)

    def __eq__(self, other):
        # Comment is not the subject of comparison
        return (
            (self._method is other.method)
            and (self._interval == other.interval)
            and (self._axis == other.axis)
            and (self._where == other.where)
        )

    def __ne__(self, other):
        return not (self == other)

    @property
    def method(self) -> _Method:
        return self._method

    @property
    def axis(self) -> str | tuple[str, ...] | None:
        return self._axis

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def comment(self) -> str:
        return self._comment

    @property
    def where(self) -> str:
        return self._where

    @property
    def numpy_operator(self) -> Callable:
        return self._METHOD_MAP[self._method].numpy_func

    @property
    def dask_operator(self) -> Callable:
        return self._METHOD_MAP[self._method].dask_func

    @classmethod
    def parse(cls, val: str | None) -> Self:
        # FIXME: These examples do not work:
        # * `val = 'lon: maximum time: mean'`,
        # * `val = 'time: mean lon: maximum'`,
        # * `val = 'lat: lon: standard_deviation (interval: 0.1 degree_N interval: 0.2 degree_E)'`.
        if val is None:
            return None
        interval_start_idx = (
            comment_start_idx
        ) = where_start_idx = par_open_index = par_close_index = np.nan
        if "interval:" in val:
            interval_start_idx = val.find("interval:")
        if "comment:" in val:
            comment_start_idx = val.find("comment:")
        if "where" in val:
            where_start_idx = val.find("where")
        if "(" in val:
            par_open_index = val.find("(")
        if ")" in val:
            par_close_index = val.find(")")

        where_val = interval_val = comment_val = ''
        idx_list = [
            interval_start_idx,
            comment_start_idx,
            where_start_idx,
            par_open_index,
            par_close_index,
        ]

        if np.isnan(idx_list).all():
            *axis, method = val.split(": ")
        else:
            *axis, method = (
                item.strip()
                for item in val[: int(np.nanmin(idx_list))].split(": ")
            )

        if not np.isnan(where_start_idx):
            # The case like `time: max where land`
            if not np.isnan(par_open_index):
                where_val = val[where_start_idx + 6 : par_open_index].strip()
            else:
                where_val = val[where_start_idx + 6 :].strip()
        if not np.isnan(interval_start_idx):
            # The case like `time : max (interval: 1hr comment: aaa)`
            # or `time : max (interval: 1hr)`
            interval_ends = int(
                np.nanmin([comment_start_idx, par_close_index])
            )
            if not np.isnan(interval_ends):
                interval_val = val[
                    interval_start_idx + 10 : interval_ends
                ].strip()
            else:
                interval_val = val[interval_start_idx + 10 :].strip()
        if not np.isnan(comment_start_idx):
            comment_val = val[comment_start_idx + 9 : par_close_index].strip()
        return cls(
            method=_Method(method),
            axis=axis,
            interval=interval_val,
            comment=comment_val,
            where=where_val
        )

    def __str__(self) -> str:
        res_str = str(self._method)
        if self._axis:
            res_str = ": ".join([*self._axis, res_str])
        if self._where:
            res_str = " ".join([res_str, f"where {self._where}"])
        if self._interval and self._comment:
            res_str = " ".join(
                [
                    res_str,
                    f"(interval: {self._interval} comment: {self._comment})",
                ]
            )
        else:
            if self._interval:
                res_str = " ".join([res_str, f"(interval: {self._interval})"])
            if self._comment:
                res_str = " ".join([res_str, f"({self._comment})"])

        return res_str
