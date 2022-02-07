import os
import timeit

import numpy as np
import pytest
from dask.delayed import Delayed

import geokube.utils.exceptions as ex
from geokube.backend.netcdf import open_dataset
