import logging
import os
import timeit
from numbers import Number

import numpy as np
import shutil

RES_PATH = os.path.join("tests", "resources", "__res.nc")
RES_DIR = os.path.join("tests", "resources", "__res")


def compare_dicts(d1, d2, exclude_d1=None, exclude_d2=None):
    exclude_d1 = (
        list(np.array(exclude_d1, ndmin=1)) if exclude_d1 is not None else []
    )
    exclude_d2 = (
        list(np.array(exclude_d2, ndmin=1)) if exclude_d2 is not None else []
    )
    for ek in set(d1.keys()):
        if ek in exclude_d1:
            continue
        assert ek in d2, f"{ek} not present in d2"
        if d1[ek] is None:
            assert d2[ek] is None
        if isinstance(d1[ek], Number) and np.isnan(d1[ek]):
            assert np.isnan(d2[ek])
        else:
            assert ek in d2
            assert d1[ek] == d2[ek], f"Key: {ek}. {d1[ek]} != {d2[ek]}"
    for ek in set(d2.keys()) - set(d1.keys()) - set(exclude_d2):
        assert ek in d1, f"{ek} not present in d1"
        if d2[ek] is None:
            assert d1[ek] is None
        if isinstance(d2[ek], Number) and np.isnan(d2[ek]):
            assert np.isnan(d1[ek])
        else:
            assert d1[ek] == d2[ek], f"Key: {ek}. {d1[ek]} != {d2[ek]}"


def clear_test_res():
    for f in [RES_DIR, RES_DIR]:
        try:
            shutil.rmtree(f)
        except:
            pass


class TimeCounter:
    def __init__(self, print=False, log=False):
        self.__print = print
        self.__log = log
        if self.__log:
            self.logger = logging.getLogger("TimeCounter")

    def __enter__(self):
        self.__start_time = timeit.timeit()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = timeit.timeit() - self.__start_time
        if self.__print:
            print(f"Execution took: {self.execution_time} msec")
        if self.__log:
            self.logger.info(f"Execution took: {self.execution_time} msec")

    @property
    def exec_time(self):
        return self.execution_time
