from __future__ import annotations

__all__ = (
    "open_datacube",
    "open_dataset",
    "build_metadata_cache",
)

import glob
import hashlib
import logging
import os
from string import Formatter
from typing import Any, Hashable, Mapping, Optional, Sequence

import dask
import pandas as pd
import xarray as xr
import rioxarray

from geokube.utils.format_parsing import reverse_format

import geokube.backend.base
import geokube.core.datacube
import geokube.core.dataset
from geokube.backend import _cache
from geokube.core.errs import CacheNotExist
from geokube.utils.hcube_logger import HCubeLogger

LOG = HCubeLogger(name="netcdf.py")

FILES_COL = geokube.core.dataset.Dataset.FILES_COL
DATACUBE_COL = geokube.core.dataset.Dataset.DATACUBE_COL


def _get_engine(path: list | str):
    if isinstance(path, list):
        if len(path) > 0:
            path = path[0]
        else:
            raise ValueError("empty path list provided!")
    if isinstance(path, str):
        _, ext = os.path.splitext(path)
    else:
        raise TypeError(f"unsupported path type: `{type(path)}`")
    if ext == ".tif":
        return "rasterio"
    elif ext == ".nc":
        return "netcdf4"
    elif ext == ".jp2":
        return "rasterio"
    elif ext == ".zarr":
        return "zarr"
    else:
        valid = sorted(xr.backends.list_engines())
        raise ValueError(
            f"No engine is associated with the path/extension `{ext or path}`."
            " Pass one explicitly via `engine=...`."
            f" Installed engines: {valid}"
        )


def _is_glob(path) -> bool:
    return isinstance(path, str) and any(c in path for c in "*?[")


def _open_raw(path, *, engine, multi, **kwargs):
    """Open a resource directly (no caching), lazy/dask-backed."""
    engine = engine or _get_engine(path)
    if engine in ("netcdf4", "zarr"):
        kwargs.setdefault("decode_coords", "all")
    if multi:
        return xr.open_mfdataset(path, engine=engine, **kwargs)
    # Single resource: open_mfdataset does not handle a lone store well. Keep it
    # lazy/dask-backed, as open_mfdataset would have been.
    kwargs.setdefault("chunks", {})
    return xr.open_dataset(path, engine=engine, **kwargs)


def _kerchunk_covers(raw, sample_file, engine) -> bool:
    """True if the kerchunk dataset retains the sample file's data variables.

    kerchunk can silently drop a variable it fails to translate; we verify against
    a plain open of one source file and reject the cache if any data variable is
    missing.
    """
    try:
        sample = _open_raw(sample_file, engine=engine, multi=False)
    except Exception:
        return True  # cannot verify -> do not block
    return set(sample.data_vars).issubset(set(raw.data_vars))


def _check_not_legacy(path) -> None:
    """Reject a regular file where a cache *directory* is expected (legacy pickle)."""
    if os.path.isfile(path):
        raise _cache.LegacyCacheFileError(
            f"metadata_cache_path is now a directory, but a regular file was found"
            f" at `{path}` (likely a legacy pickle cache): remove it or pass a"
            f" directory."
        )


# ============================================================ READER (API side)
# open_datacube / open_dataset only READ a cache published by the catalog. They
# never write, glob, stat or rebuild. A missing cache raises CacheNotExist.

def open_datacube(
    path: str,
    id_pattern: Optional[str] = None,
    mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    metadata_caching: bool = False,
    metadata_cache_path: str = None,
    concat_dims: Optional[Sequence[str]] = None,  # build-time; accepted for symmetry
    identical_dims: Optional[Sequence[str]] = None,
    **kwargs,  # optional kw args for the xarray opener
) -> geokube.core.datacube.DataCube:
    """Open one :class:`DataCube` from a single resource, a list of files or a glob.

    With ``metadata_caching=True`` (multi-file paths only) this is a **read-only**
    operation: it loads the kerchunk reference published under the
    ``metadata_cache_path`` *directory* by the catalog (see
    :func:`build_metadata_cache`) and never writes. If the cache is absent it raises
    :class:`~geokube.core.errs.CacheNotExist`.

    Without caching it opens the data directly (lazy/dask-backed).
    """
    engine = kwargs.pop("engine", None)
    multi = isinstance(path, (list, tuple)) or _is_glob(path)

    if metadata_caching and multi:
        return _read_datacube_cache(
            metadata_cache_path, id_pattern=id_pattern, mapping=mapping
        )

    raw = _open_raw(path, engine=engine, multi=multi, **kwargs)
    return geokube.core.datacube.DataCube.from_xarray(
        raw, id_pattern=id_pattern, mapping=mapping
    )


def _read_datacube_cache(
    metadata_cache_path, *, id_pattern, mapping
) -> geokube.core.datacube.DataCube:
    from geokube.backend import _kerchunk

    if metadata_cache_path is None:
        raise ValueError(
            "If `metadata_caching` is True, `metadata_cache_path` must be provided!"
        )
    _check_not_legacy(metadata_cache_path)
    payload = _kerchunk.load_store(metadata_cache_path)
    if payload is None:
        raise CacheNotExist(
            f"No metadata cache found at `{metadata_cache_path}`. It must be built"
            " by the catalog via build_metadata_cache() before read-only access."
        )
    raw = _kerchunk.open_store(payload)
    return geokube.core.datacube.DataCube.from_xarray(
        raw, id_pattern=id_pattern, mapping=mapping
    )


def open_dataset(
    path: str,
    pattern: str,
    id_pattern: Optional[str] = None,
    mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    metadata_caching: bool = False,
    metadata_cache_path: str = None,
    ds_attr_mapping: Mapping[
        Hashable, Any
    ] = None,  # dataset attributes mapping - TBA
    ncvars_mapping: Mapping[Hashable, Any] = None,  # netcdf variables mapping - TBA
    delay_read_cubes: bool = False,
    load_files_on_persistance: bool = True,
    concat_dims: Optional[Sequence[str]] = None,  # build-time; forwarded to cubes
    identical_dims: Optional[Sequence[str]] = None,
    **kwargs,  # optional kw args for the xarray opener
) -> geokube.core.dataset.Dataset:
    """Open a catalog :class:`Dataset` from files grouped by a filename ``pattern``.

    With ``metadata_caching=True`` this is **read-only**: it loads the file-index
    (``index.json``) and the per-group kerchunk references published by the catalog
    under the ``metadata_cache_path`` *directory*. It never writes. A missing cache
    raises :class:`~geokube.core.errs.CacheNotExist`; (re)building is the catalog's
    job via :func:`build_metadata_cache`.
    """
    ds_attr_names = _get_ds_attrs_names(pattern)

    if metadata_caching:
        if metadata_cache_path is None:
            raise ValueError(
                "If `metadata_caching` set to True, `metadata_cache_path`"
                " argument needs to be provided!"
            )
        _check_not_legacy(metadata_cache_path)
        index = _cache.read_json(
            os.path.join(metadata_cache_path, _cache.INDEX_FILE)
        )
        if index is None:
            raise CacheNotExist(
                f"No metadata cache found at `{metadata_cache_path}`. It must be"
                " built by the catalog via build_metadata_cache() before read-only"
                " access."
            )
        attrs = index.get("attrs", ds_attr_names)
        df = _cache.records_to_indexed_df(index["records"], attrs, FILES_COL)
        df = _attach_datacubes(
            df,
            id_pattern=id_pattern,
            mapping=mapping,
            delay_read_cubes=delay_read_cubes,
            load_files_on_persistance=load_files_on_persistance,
            cubes_cache_dir=os.path.join(metadata_cache_path, _cache.CUBES_DIR),
            concat_dims=concat_dims,
            identical_dims=identical_dims,
            **kwargs,
        )
        return geokube.core.dataset.Dataset(
            hcubes=df.reset_index(),
            load_files_on_persistance=load_files_on_persistance,
        )

    # caching disabled: read files in path directly
    files = glob.glob(path)
    df = _get_df_from_files_list(files, pattern, ds_attr_names)
    df = _attach_datacubes(
        df,
        id_pattern=id_pattern,
        mapping=mapping,
        delay_read_cubes=delay_read_cubes,
        load_files_on_persistance=load_files_on_persistance,
        cubes_cache_dir=None,
        concat_dims=concat_dims,
        identical_dims=identical_dims,
        **kwargs,
    )
    return geokube.core.dataset.Dataset(
        hcubes=df.reset_index(),
        load_files_on_persistance=load_files_on_persistance,
    )


# ============================================================ WRITER (catalog)
# build_metadata_cache is the ONLY component that writes/invalidates the cache.
# It is meant to run out-of-band in the catalog (which has write permission); the
# read-only API only consumes what it publishes.

def build_metadata_cache(
    path: str,
    pattern: Optional[str] = None,
    *,
    metadata_cache_path: str,
    id_pattern: Optional[str] = None,
    mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    combine: str = "by_coords",
    concat_dim: Optional[str] = None,
    engine: Optional[str] = None,
    concat_dims: Optional[Sequence[str]] = None,  # deprecated; ignored
    identical_dims: Optional[Sequence[str]] = None,  # deprecated; ignored
    **kwargs,
) -> dict:
    """Build / refresh the metadata cache (catalog/writer entrypoint).

    * ``pattern is None`` -> single combined cube: caches the kerchunk reference for
      the resolved file list/glob under ``metadata_cache_path``.
    * ``pattern`` given -> catalog: builds the file-index and a per-group kerchunk
      reference; the cube stores are written first and ``index.json`` last, so a
      concurrent read-only reader never sees an index pointing at missing stores.

    ``combine`` selects how the per-file references are recombined at open time,
    mirroring ``open_mfdataset``: ``"by_coords"`` (default; orders by coordinate, so
    it tolerates out-of-order filenames and mixed NetCDF3/NetCDF4) or ``"nested"``
    along ``concat_dim`` for a bare index axis lacking a coordinate (e.g. NSIDC
    ``tdim``). The spec is stored in the cache and replayed by the reader, which
    therefore needs no combine argument. (``concat_dims``/``identical_dims`` are
    accepted for backward compatibility but ignored.)

    Invalidation is automatic and **incremental** (per-file ``(mtime, size)``
    manifest): unchanged files reuse their cached references; only changed ones are
    re-read. Groups whose format kerchunk cannot reference (e.g. GeoTIFF) are
    **skipped** (not cached) and reported in the summary.

    Returns ``{"groups": int, "built": int, "skipped": [..]}``.
    """
    if metadata_cache_path is None:
        raise ValueError("`metadata_cache_path` must be provided.")

    if pattern is None:
        files = sorted(glob.glob(path)) if _is_glob(path) else list(path)
        if not files:
            raise ValueError("No files found for the provided path!")
        cache_dir = _cache.ensure_cache_dir(metadata_cache_path)
        ok = _build_datacube_cache(
            files,
            cache_dir,
            combine=combine,
            concat_dim=concat_dim,
            engine=engine,
        )
        return {
            "groups": 1,
            "built": 1 if ok else 0,
            "skipped": [] if ok else [str(metadata_cache_path)],
        }

    cache_dir = _cache.ensure_cache_dir(metadata_cache_path)
    ds_attr_names = _get_ds_attrs_names(pattern)
    files = sorted(glob.glob(path))
    df = _get_df_from_files_list(files, pattern, ds_attr_names)
    cubes_dir = os.path.join(cache_dir, _cache.CUBES_DIR)
    built, skipped = 0, []
    for i in df.index:
        ok = _build_datacube_cache(
            df[FILES_COL][i],
            _cube_cache_dir(cubes_dir, i),
            combine=combine,
            concat_dim=concat_dim,
            engine=engine,
        )
        if ok:
            built += 1
        else:
            skipped.append(repr(i))
    # Publish the index LAST: it now references only already-written cube stores.
    _cache.write_json(
        os.path.join(cache_dir, _cache.INDEX_FILE),
        {
            "attrs": ds_attr_names,
            "records": _cache.index_to_records(df, ds_attr_names, FILES_COL),
        },
    )
    return {"groups": int(len(df.index)), "built": built, "skipped": skipped}


def _build_datacube_cache(
    files, cache_dir, *, combine, concat_dim, engine
) -> bool:
    """Build/refresh one cube's kerchunk store under ``cache_dir`` (incremental).

    Returns ``True`` if a valid store is cached, ``False`` if the files are not
    kerchunk-referenceable or kerchunk would drop variables (nothing is published).
    """
    from geokube.backend import _kerchunk

    cache_dir = _cache.ensure_cache_dir(cache_dir)
    context = {
        "kind": "datacube",
        "combine": combine,
        "concat_dim": concat_dim,
        "kerchunk_version": _kerchunk.KERCHUNK_VERSION,
    }
    current = _cache.build_manifest(files, context=context)
    manifest_path = os.path.join(cache_dir, _cache.MANIFEST_FILE)
    cached = _cache.read_json(manifest_path)

    # Already up to date -> no-op (incremental skip).
    if _cache.manifest_matches(cached, current) and (
        _kerchunk.load_store(cache_dir) is not None
    ):
        return True

    reuse = ()
    if _cache.context_matches(cached, current):
        diff = _cache.manifest_diff(cached, current)
        stale = set(diff["added"]) | set(diff["changed"])
        reuse = [k for k in current["files"] if k not in stale]
    payload = _kerchunk.cached_build_store(
        files,
        cache_dir,
        reuse_keys=reuse,
        combine=combine,
        concat_dim=concat_dim,
    )
    if payload is None:
        return False
    raw = _kerchunk.open_store(payload)
    if not _kerchunk_covers(raw, files[0], engine):
        return False
    # Manifest written last (after store.json): marks the store as valid.
    _cache.write_json(manifest_path, current)
    return True


# -------------------------------------------------------------- shared helpers

def _get_ds_attrs_names(pattern):
    fmt = Formatter()
    # get the dataset attrs from the pattern
    ds_attr_names = [i[1] for i in fmt.parse(pattern) if i[1]]
    return ds_attr_names


def _get_df_from_files_list(files, pattern, ds_attr_names):
    l = []
    for f in files:
        d = reverse_format(pattern, f)
        d[FILES_COL] = f
        l.append(d)
    df = pd.DataFrame(l)
    if len(l) == 0:
        raise ValueError(f"No files found for the provided path!")
    # unique index for each dataset attribute combos - we create a list of files
    df = df.groupby(ds_attr_names)[FILES_COL].apply(list).reset_index()
    df = df.set_index(ds_attr_names)
    return df


def _cube_cache_dir(cubes_cache_dir, group_key):
    """Stable per-group cache sub-directory (keyed by the attribute combo)."""
    h = hashlib.sha1(repr(group_key).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cubes_cache_dir, h)


def _attach_datacubes(
    df,
    *,
    id_pattern,
    mapping,
    delay_read_cubes,
    load_files_on_persistance,
    cubes_cache_dir=None,
    concat_dims=None,
    identical_dims=None,
    **kwargs,
):
    """Populate ``DATACUBE_COL`` from ``FILES_COL`` (shared read/cold path).

    When ``cubes_cache_dir`` is given each group's DataCube is opened (read-only)
    from its kerchunk cache; otherwise the files are opened directly.
    """
    if not load_files_on_persistance:
        df[DATACUBE_COL] = None
        return df
    cubes = []
    for i in df.index:
        files_i = df[FILES_COL][i]
        cube_kwargs = dict(id_pattern=id_pattern, mapping=mapping, **kwargs)
        if cubes_cache_dir is not None:
            cube_kwargs.update(
                metadata_caching=True,
                metadata_cache_path=_cube_cache_dir(cubes_cache_dir, i),
                concat_dims=concat_dims,
                identical_dims=identical_dims,
            )
        if delay_read_cubes:
            cubes.append(dask.delayed(open_datacube)(path=files_i, **cube_kwargs))
        else:
            cubes.append(open_datacube(path=files_i, **cube_kwargs))
    df[DATACUBE_COL] = cubes
    return df
