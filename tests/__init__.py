import logging
import timeit
import os

RES_PATH = os.path.join("tests", "resources", "__res.nc")


def clear_test_res():
    try:
        os.remove(RES_PATH)
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
