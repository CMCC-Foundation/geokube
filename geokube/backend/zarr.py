import os
from typing import Any, List, Mapping, Optional, Tuple, Union

import dask.bag as db
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from intake.source.utils import reverse_format
from zarr.storage import ContainsArrayError

import geokube.utils.exceptions as ex
from geokube import LOGGER_NAME
from geokube.backend.base import BaseOpener
from geokube.core.container import Container
from geokube.core.cube import Cube
from geokube.core.dataset_metadata import DatasetMetadata


class _ZarrOpenManager(BaseOpener):
    @staticmethod
    def _wrap_reverse_format_zarr(pattern: str, f: str) -> Optional[Mapping]:
        try:
            return reverse_format(pattern, f)
        except ValueError:
            return None

    def _process_single_cube(
        self, group: str, s3map: s3fs.S3Map, pattern: str
    ) -> Optional[Cube]:
        fields = _ZarrOpenManager._wrap_reverse_format_zarr(pattern, group)
        if fields is None:
            return None
        try:
            tmp_ds = xr.open_zarr(store=s3map, group=group)
        except ContainsArrayError:
            return None
        self.set_attributes(tmp_ds)
        return Cube(data=tmp_ds, files=group), fields

    def _open_zarr_hypercube(
        self,
        root: str,
        pattern: str,
        zarr_open_kwargs: Mapping[str, Any] = None,
        **open_kwargs,
    ) -> Container:
        import logging

        logger = logging.getLogger(LOGGER_NAME)
        logger.debug(
            "Connecting to S3 storage at address:"
            " http://51.159.24.124:9000 ..."
        )
        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={
                "region_name": "eu-west-1",
                "endpoint_url": "http://51.159.24.124:9000",
            },
        )
        s3map = s3fs.S3Map(root=root, s3=fs, check=True)
        unique_addresses = np.unique(
            [os.path.dirname(k) for k in s3map.keys() if not k.startswith(".")]
        )
        logger.debug(
            f"Taking unique keys of S3Map...Found: {len(unique_addresses)}"
        )
        bag = db.from_sequence(
            unique_addresses, npartitions=open_kwargs.get("npartitions", 30)
        )
        # ##### ITERATIVE FRAGMENT FOR DEBUG PURPOSES ##### #
        # for i in unique_addresses:
        #     import pdb;pdb.set_trace()
        #     self._process_single_cube(i, s3map=s3map, pattern=pattern)
        # ################################################### #
        results = bag.map(
            self._process_single_cube, s3map=s3map, pattern=pattern
        ).compute()
        results = [_ for _ in results if _ is not None]
        if not len(results):
            logger.error(
                f"No data found for root: {root}  and pattern: {pattern}!"
            )
            raise ValueError(
                f"No data found for root: {root}  and pattern: {pattern}!"
            )
        cubes, fields = zip(*results)
        cubes = pd.Series(cubes)
        fields = pd.DataFrame(fields)
        dataset_name, product_type, fields = self._extract_dataset_metadata(
            fields
        )
        if not len(fields.columns):
            fields = None
        return Container(
            cubes_df=cubes,
            fields_df=fields,
            fields_df_metadata=self.fields_mapping,
            dataset_metadata=DatasetMetadata.parse(
                {}, dataset_name=dataset_name, product_type=product_type
            ),
            query_limit_gb=None,
        )
