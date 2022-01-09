import logging
import os
from typing import Any, Mapping, NoReturn, Optional

import geokube.utils.exceptions as ex
from geokube import LOGGER_NAME
from geokube.backend.base import BaseOpener
from geokube.core.container import Container


class _ZipOpenManager(BaseOpener):
    def _open_zip_hypercube(
        self,
        zipfilename: str,
        pattern: str,
        zip_open_kwargs: Mapping[str, Any] = None,
        **open_kwargs,
    ) -> Container:
        from functools import partial
        from zipfile import ZipFile

        from geokube.backend.netcdf import _NetCDFOpenManager

        logger = logging.getLogger(LOGGER_NAME)
        unzip_target = zipfilename.split(".zip")[0] + "_unzipped"
        logger.info(f"Unzipping to temporary directory: {unzip_target}")
        with ZipFile(zipfilename) as zip_input:
            zip_input.extractall(unzip_target)
        files = os.listdir(unzip_target)
        if not len(files):
            logger.warning("No files found in unzipped directory")
            raise ex.HCubeValueError("No files found in unzipped directory")
        pattern = os.path.join(unzip_target, pattern)
        hypercube = _NetCDFOpenManager(
            variable_mapping=self.variable_mapping,
            coordinate_mapping=self.coordinate_mapping,
        )._open_netcdf_hypercube(pattern=pattern)
        hypercube.close_process = partial(
            _ZipOpenManager.remove_dir, unzip_target=unzip_target
        )
        logger.info("Opening zip hypercube finished successfully")
        return hypercube

    @staticmethod
    def remove_dir(unzip_target) -> NoReturn:
        import shutil

        logger = logging.getLogger(LOGGER_NAME)
        logger.info(f"Attempt to clear temporary directory: {unzip_target} ...")
        shutil.rmtree(unzip_target, ignore_errors=True)
        logger.info("Removing finished successfully")
