import os
import numpy as np
from dask.delayed import Delayed
import pytest
import timeit

import geokube.utils.exceptions as ex
from geokube.backend.netcdf import open_dataset
