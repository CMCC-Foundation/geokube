from __future__ import annotations

from typing import Iterable, Union

import numpy as np

from .enums import MethodType


class CellMethod:
    # Based on  https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch07s03.html
    # Currently supports non-combined cell methods
    # TODO: extend for combined ones and special variables, like `lat: lon: standard_deviation`, `area: mean`, etc.
    def __init__(
        self,
        method: MethodType = None,
        interval: str = None,  # Timedelta/Interval type??
        axis: Union[
            str, Iterable[str]
        ] = None,  #  # axis on which is defined (could be more than one axis??)
        comment: str = None,
        where: str = None,
    ) -> None:
        self.method = method if method is not None else MethodType.UNDEFINED
        self.axis = axis
        self.interval = interval
        self.comment = comment
        self.where = where

    def __eq__(self, other):
        # Comment is not the subject of comparison
        return (
            (self.method is other.method)
            and (self.interval == other.interval)
            and (self.axis == other.axis)
            and (self.where == other.where)
        )

    def __ne__(self, other):
        return not (self == other)

    @classmethod
    def parse(cls, val: str) -> "CellMethod":
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

        where_val = interval_val = comment_val = None
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
        return CellMethod(
            method=MethodType(method),
            interval=interval_val,
            axis=axis,
            where=where_val,
            comment=comment_val,
        )

    def __str__(self) -> str:
        res_str = str(self.method)
        if self.axis is not None:
            res_str = ": ".join([*self.axis, res_str])
        if self.where is not None:
            res_str = " ".join([res_str, f"where {self.where}"])
        if (self.interval is not None) and (self.comment is not None):
            res_str = " ".join(
                [
                    res_str,
                    f"(interval: {self.interval} comment: {self.comment})",
                ]
            )
        else:
            if self.interval is not None:
                res_str = " ".join([res_str, f"(interval: {self.interval})"])
            if self.comment is not None:
                res_str = " ".join([res_str, f"({self.comment})"])

        return res_str
