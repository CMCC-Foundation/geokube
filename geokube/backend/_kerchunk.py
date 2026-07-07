"""VirtualiZarr metadata store for lazy multi-file ``DataCube`` caching.

Opening hundreds/thousands of NetCDF files with ``xr.open_mfdataset`` can take
hours: the cost is the *open* (per-file metadata read + coordinate alignment), not
the data processing. This module persists that per-file metadata as a **VirtualiZarr
manifest** (serialized in the kerchunk reference format), so subsequent opens are
cheap, portable and robust across library versions.

Design:

* Each file is opened as a per-file **virtual dataset** with the VirtualiZarr parser
  matching its on-disk format (HDF5/NetCDF4 -> :class:`HDFParser`; NetCDF3-classic ->
  :class:`NetCDF3Parser`), chosen by magic-byte sniffing. Dimension coordinates are
  loaded eagerly (real numpy) while data variables stay virtual (``ManifestArray``).
  The per-file manifest is persisted under ``files/`` (kerchunk JSON) for incremental
  rebuilds.
* The per-file virtual datasets are **partitioned by encoding signature** (dtype,
  chunks, codecs per variable) and each homogeneous partition is consolidated into a
  single virtual dataset with a **vectorized manifest concat** (``xr.concat`` /
  ``xr.combine_by_coords`` — orders of magnitude faster than the old
  ``MultiZarrToZarr`` per-chunk loop), then written as a **kerchunk PARQUET** manifest
  under ``parts/``. A single Zarr array cannot describe heterogeneous layouts, so e.g.
  contiguous/uncompressed NetCDF3 and chunked/compressed NetCDF4 fall into different
  partitions; the (few) partitions are recombined at open time. With one partition
  (the common case) open is a single lazy reference — **instant**, no combine.
* ``by_coords`` orders by coordinate (tolerates out-of-order files and mixed formats);
  ``nested`` stacks in input/file order along an explicit ``concat_dim`` (bare index
  axis with no coordinate). Both consolidate at build time; the combine spec is
  persisted in ``store.json`` and replayed by the reader only when >1 partition exists.

Parquet manifests load lazily/vectorized via pyarrow, so opening a consolidated store
reads only the array metadata (and the eagerly-loaded coordinates), never the data
chunks. The store is produced by :func:`cached_build_store` and reopened by
:func:`open_store` (``cache_dir`` is injected by :func:`load_store`).
"""
from __future__ import annotations

__all__ = [
    "VZ_VERSION",
    "STORE_SCHEMA_VERSION",
    "FILES_SUBDIR",
    "PARTS_SUBDIR",
    "STORE_FILE",
    "detect_format",
    "is_referenceable",
    "reference_one",
    "open_reference",
    "cached_build_store",
    "load_store",
    "open_store",
]

import base64
import hashlib
import json
import math
import os
import shutil
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import fsspec
import xarray as xr

import virtualizarr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import NetCDF3Parser, HDFParser, KerchunkJSONParser
try:  # virtualizarr>=2.7 moved the registry; keep a fallback for older layouts
    from obspec_utils.registry import ObjectStoreRegistry
except Exception:  # pragma: no cover
    from virtualizarr.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from geokube.backend import _cache
from geokube.backend._progress import dask_progress, is_distributed, progress_iter
from geokube.utils.attrs_encoding import is_undecodable_time_unit
from geokube.utils.format_parsing import _make_path_posix
from geokube.utils.hcube_logger import HCubeLogger

LOG = HCubeLogger(name="_kerchunk.py")

VZ_VERSION = virtualizarr.__version__

# Bump when the on-disk store shape changes; old stores then read as a miss (the
# catalog rebuilds them). Schema 3 = VirtualiZarr per-partition parquet manifests;
# schema 4 adds the `combine_plan` (merge-aware recombination: partitions sharing a
# record axis are merged, distinct-axis groups concatenated). Schema 5 adds
# `passthrough` (per-partition: variables whose native chunk spans the whole array
# are excluded from the kerchunk reference entirely and reopened directly from their
# source file(s) at read time -- see `_detect_passthrough` / `_splice_passthrough`).
STORE_SCHEMA_VERSION = 5

# Cache-directory layout for the per-cube store.
FILES_SUBDIR = "files"      # one per-file kerchunk JSON manifest (incremental rebuild)
PARTS_SUBDIR = "parts"      # one parquet manifest directory per encoding partition
STORE_FILE = "store.json"   # partition index + combine spec (relative parquet paths)

# Combine strategies persisted in the store and replayed at open time.
COMBINE_BY_COORDS = "by_coords"
COMBINE_NESTED = "nested"

# On-disk format for the consolidated per-partition manifest. ``parquet`` (a directory,
# loaded lazily/vectorized via pyarrow) is the production default for instant opens at
# scale; ``json`` (a single file) is a simpler fallback for small cubes.
_COMBINED_FORMAT = "parquet"

# Sentinel default: defer scheduler selection to dask, like xarray does.
SCHEDULER_AUTO = "auto"

# xarray opener kwargs honored on the cached read path. They are forwarded to
# ``xr.open_dataset(mapper, engine="zarr", ...)`` in :func:`open_reference`, persisted
# in the store at build time (:func:`_assemble_store`) and replayed by the reader, so
# the cache stays transparent w.r.t. how the data is opened. ``combine``/``concat_dim``/
# ``engine``/``scheduler`` are NOT here: they are build/combine params, not read options
# (``engine`` is fixed to ``zarr`` for the reference store). ``chunks`` is the lever that
# bounds the dask task graph at scale (O(#blocks) instead of O(#on-disk-chunks)).
_FORWARDED_OPEN_KWARGS = frozenset({
    "chunks", "decode_cf", "decode_times", "decode_timedelta", "decode_coords",
    "mask_and_scale", "concat_characters", "use_cftime", "drop_variables",
})

# Hardcoded defaults applied when an option is neither persisted nor passed at read time.
_OPEN_DEFAULTS = {"decode_coords": "all", "chunks": {}}

# Native xarray engine used to reopen a "passthrough" variable directly from its
# source file(s), bypassing the kerchunk reference layer entirely (see
# ``_detect_passthrough``/``_splice_passthrough`` below). ``detect_format``'s
# "hdf5"/"netcdf3" strings are NOT valid xarray engine names on their own --
# ``h5netcdf`` cannot even open a NetCDF3-classic file -- and ``netcdf4`` (the full
# netCDF-C library) reads both formats with more dask/thread-safe locking than the
# mmap-based ``scipy`` backend, which xarray itself flags as fragile under dask.
_PASSTHROUGH_ENGINE = {"hdf5": "h5netcdf", "netcdf3": "netcdf4"}

# Default cap on how many source files a single passthrough variable may span within
# one partition before falling back to today's plain (single giant chunk) kerchunk
# reference for it. Reopening N files natively at read time is cheap when N is small
# (the motivating case is N=1: one contiguous file per model/cube); an unbounded N
# would silently reintroduce the per-file open cost this cache exists to avoid.
DEFAULT_PASSTHROUGH_MAX_FILES = 8

# Number of native on-disk reference chunks along a dimension above which a variable
# that is NOT a passthrough candidate (i.e. genuinely fine-grained -- e.g. a NetCDF3
# *record* variable, referenced one kerchunk chunk per record by
# ``kerchunk.netCDF3.NetCDF3ToZarr``) gets a build-time warning if the effective
# ``chunks`` open kwarg is left at the bare default. Opening O(#records) reference
# chunks with ``chunks={}`` reproduces the "OOM in apertura a scala" graph blow-up
# this cache already fixed once, just from the opposite (too-fine, not too-coarse)
# native chunking.
_MANY_CHUNKS_WARN_THRESHOLD = 2000


def _var_native_chunks(vds: xr.Dataset, name: str) -> Optional[List[int]]:
    """The native/manifest chunk shape for a lazy data variable, or ``None`` if it
    carries no manifest metadata at all (e.g. an eagerly-loaded coordinate)."""
    meta = getattr(vds.variables[name].data, "metadata", None)
    if meta is None:
        return None
    return list(getattr(meta, "chunks", []) or [])


def _is_contiguous_chunk(chunks: Optional[Sequence[int]], shape: Sequence[int]) -> bool:
    """True if ``chunks`` denotes a single chunk spanning the whole array -- no chunk
    grid at all (``chunks`` empty) or an explicit chunk shape equal to the full
    extent. HDF5/NetCDF4 contiguous storage and NetCDF3-classic non-record variables
    both surface this way: VirtualiZarr/kerchunk report one chunk shaped like the
    whole variable (``kerchunk.netCDF3.NetCDF3ToZarr.translate`` hardcodes
    ``chunks=shape`` for non-record variables -- a literal ``# TODO: chance to
    sub-chunk`` left unimplemented in that library)."""
    return not chunks or list(chunks) == list(shape)


def _detect_passthrough(vds: xr.Dataset, *, enabled: bool) -> dict:
    """Data variables in a per-file virtual dataset whose native chunk spans the
    whole variable (:func:`_is_contiguous_chunk`) -- candidates to skip the kerchunk
    reference entirely and be reopened directly from source at read time
    (:func:`_splice_passthrough`). Returns ``{name: native_chunks}``.

    ``vds`` is one representative file from an encoding-signature group
    (:func:`_partition`): the signature folds the native chunk shape into the
    grouping key, so every file sharing it has an identical per-file chunk/shape for
    a given variable -- checking one file stands for the whole group, independent of
    how many of them later get consolidated into one manifest by :func:`_consolidate`.
    """
    if not enabled:
        return {}
    out = {}
    for name in vds.data_vars:
        chunks = _var_native_chunks(vds, name)
        if chunks is None:
            continue
        if _is_contiguous_chunk(chunks, vds[name].shape):
            out[name] = chunks
    return out


def _warn_if_many_small_chunks(
    vds: xr.Dataset, dim: Optional[str], n_files: int,
    open_kwargs: Optional[Mapping], passthrough_vars: Mapping,
) -> None:
    """Log a build-time warning when a (non-passthrough) variable's native chunking
    along ``dim`` would yield so many reference chunks that opening with the bare
    ``chunks={}`` default (no coalescing requested) blows up the dask graph at open --
    the mirror-image of the contiguous-chunk problem :func:`_detect_passthrough`
    handles. Passthrough candidates are exempt: they are never opened via the
    reference path."""
    if dim is None or (open_kwargs or {}).get("chunks"):
        return  # caller already requested explicit coalescing -- nothing to warn about
    for name in vds.data_vars:
        if name in passthrough_vars:
            continue
        chunks = _var_native_chunks(vds, name)
        if not chunks or dim not in vds[name].dims:
            continue
        chunk = chunks[vds[name].dims.index(dim)]
        if not chunk:
            continue
        n_chunks = math.ceil(vds.sizes[dim] / chunk) * max(n_files, 1)
        if n_chunks > _MANY_CHUNKS_WARN_THRESHOLD:
            LOG.warn(
                f"variable `{name}` has ~{n_chunks} native reference chunks along"
                f" `{dim}` (on-disk chunk size {chunk}); opening with the default"
                ' `chunks={}` builds one dask task per chunk. Pass an explicit'
                f' `chunks={{"{dim}": N}}` (or "auto") to `build_metadata_cache`'
                " to coalesce the graph."
            )
            return


def _filter_open_kwargs(open_kwargs: Optional[Mapping]) -> dict:
    """Keep only the xarray opener kwargs safe to forward to the zarr-reference open."""
    return {
        k: v for k, v in (open_kwargs or {}).items() if k in _FORWARDED_OPEN_KWARGS
    }


def _decode_open_kwargs(open_kwargs: Optional[Mapping]) -> dict:
    """Decode-only subset (drop ``chunks``) for the build-time coordinate/scalar reads,
    which immediately materialize small 1-D/2-D arrays and so always open with ``chunks={}``."""
    return {
        k: v for k, v in _filter_open_kwargs(open_kwargs).items() if k != "chunks"
    }


# The decode-related kwargs forwarded to ``virtualizarr.open_virtual_dataset`` at build so the
# cache honors the catalog's flags — notably ``decode_times=False`` for non-CF time units like
# ``months since ...`` that xarray/cftime cannot decode — matching the non-cached open path.
# ``drop_variables`` is deliberately NOT here: ``open_virtual_dataset`` applies it via a STRICT
# ``Dataset.drop_vars`` (``errors="raise"``) that raises when a listed variable is absent from a
# given file; :func:`reference_one` applies it itself with xarray-lenient semantics instead.
_VDS_OPEN_KWARGS = frozenset({"decode_times"})


def _vds_open_kwargs(open_kwargs: Optional[Mapping]) -> dict:
    """Subset of ``open_kwargs`` that ``open_virtual_dataset`` accepts (build-time manifest)."""
    return {k: v for k, v in (open_kwargs or {}).items() if k in _VDS_OPEN_KWARGS}


_HDF5_MAGIC = b"\x89HDF"
_NETCDF3_MAGICS = (b"CDF\x01", b"CDF\x02", b"CDF\x05")


# ------------------------------------------------------------- format detection

def detect_format(path: str) -> str:
    """Return ``'hdf5'`` / ``'netcdf3'`` / ``'unknown'`` by magic bytes."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head == _HDF5_MAGIC:
        return "hdf5"
    if head in _NETCDF3_MAGICS:
        return "netcdf3"
    return "unknown"


def is_referenceable(path: str) -> bool:
    """True if VirtualiZarr can build a manifest for this file's format."""
    return detect_format(path) in ("hdf5", "netcdf3")


def _time_units_undecodable(path: str) -> bool:
    """Cheap metadata-only probe: does any variable carry a non-CF ``months``/``years since``
    time unit that xarray/cftime cannot decode?

    Opens with ``decode_times=False`` (so the probe itself never raises) and ``chunks={}``
    (reads only header metadata, no data). Used to auto-apply ``decode_times=False`` for the
    whole build so the manifest read never crashes even when the caller forgot the flag."""
    try:
        with xr.open_dataset(
            path, decode_times=False, chunks={}, decode_coords="all"
        ) as ds:
            for var in ds.variables.values():
                units = var.attrs.get("units", var.encoding.get("units"))
                calendar = var.attrs.get("calendar", var.encoding.get("calendar"))
                if is_undecodable_time_unit(units, calendar):
                    return True
    except Exception:
        return False
    return False


# ------------------------------------------------------ virtualizarr plumbing

# A single filesystem-rooted object store resolves every absolute ``file://`` URL
# (per-file manifests reference the original NetCDF byte ranges by absolute path).
_REGISTRY = ObjectStoreRegistry({"file://": LocalStore()})


def _url(path: str) -> str:
    return "file://" + os.path.abspath(path)


def _parser_for(path: str):
    fmt = detect_format(path)
    if fmt == "hdf5":
        return HDFParser()
    if fmt == "netcdf3":
        return NetCDF3Parser()
    raise ValueError(
        f"Cannot build a VirtualiZarr manifest for `{path}` (unrecognized format)."
    )


def _vz(vds: xr.Dataset):
    """Return the VirtualiZarr accessor (``.vz`` on 2.7+, ``.virtualize`` older)."""
    acc = getattr(vds, "vz", None)
    return acc if acc is not None else vds.virtualize


def _to_kerchunk(vds: xr.Dataset, path: str, fmt: str) -> None:
    _vz(vds).to_kerchunk(path, format=fmt)


def _drop_scalars(vds: xr.Dataset) -> xr.Dataset:
    """Drop 0-dim variables — VirtualiZarr cannot round-trip them through a kerchunk
    manifest (unreadable virtual scalar; empty chunk key on reload). They are handled
    out-of-band by :func:`_extract_scalars` / :func:`_reattach_scalars`."""
    names = [str(v) for v in vds.variables if vds[v].ndim == 0]
    return vds.drop_vars(names) if names else vds


def reference_one(
    path: str, *, loadable_variables=None, open_kwargs: Optional[Mapping] = None
) -> xr.Dataset:
    """Open one file as a fully-virtual per-file dataset (0-dim variables dropped).

    Everything stays a lazy ``ManifestArray`` (``loadable_variables=[]``): nothing is
    loaded eagerly, so the per-file kerchunk JSON persisted for incremental rebuilds
    re-hydrates cleanly via :func:`_reload_one` (eagerly-loaded coordinates do not
    round-trip through VirtualiZarr's kerchunk parser). The concat coordinate is
    materialized inline only once, on the consolidated partition, in
    :func:`_assemble_store`. Scalars are dropped (handled via the sidecar). This is the
    embarrassingly-parallel hot spot of a cold build.

    ``open_kwargs`` carries the catalog's decode flags; the subset ``open_virtual_dataset``
    accepts (:func:`_vds_open_kwargs`) is forwarded so the manifest build honors them — e.g.
    ``decode_times=False`` for non-CF ``months since ...`` time that xarray/cftime cannot
    decode — keeping the cache transparent w.r.t. the non-cached open.
    """
    lv = [] if loadable_variables is None else list(loadable_variables)
    vds = open_virtual_dataset(
        _url(path), registry=_REGISTRY, parser=_parser_for(path), loadable_variables=lv,
        **_vds_open_kwargs(open_kwargs),
    )
    # Apply ``drop_variables`` here with xarray-lenient semantics rather than letting
    # ``open_virtual_dataset`` do it: VirtualiZarr routes ``drop_variables`` through a strict
    # ``Dataset.drop_vars`` (``errors="raise"``) that blows up when a listed variable is absent
    # from *this* file, but the same var is often present in only some files of a dataset (e.g.
    # ``time_bnds`` in bioclimind). ``errors="ignore"`` mirrors ``xr.open_dataset`` and keeps the
    # cache transparent w.r.t. the non-cached open. The drop is post-decode either way (it does
    # not avoid a decode crash — use ``decode_times=False`` for that).
    drop = list((open_kwargs or {}).get("drop_variables") or ())
    if drop:
        vds = vds.drop_vars(drop, errors="ignore")
    return _drop_scalars(vds)


def _reload_one(
    json_path: str, *, open_kwargs: Optional[Mapping] = None
) -> Optional[xr.Dataset]:
    """Re-hydrate a fully-virtual per-file dataset from its cached kerchunk JSON.

    Forwards the same decode subset as :func:`reference_one` so an incremental rebuild that
    re-hydrates a manifest built with e.g. ``decode_times=False`` does not try (and fail) to
    decode the raw time on reload."""
    if not os.path.isfile(json_path):
        return None
    try:
        return open_virtual_dataset(
            _url(json_path), registry=_REGISTRY, parser=KerchunkJSONParser(),
            loadable_variables=[], **_vds_open_kwargs(open_kwargs),
        )
    except Exception:
        return None


# ------------------------------------------------------------------ signatures

def _signature(vds: xr.Dataset) -> str:
    """Stable key for a virtual dataset's encoding (dtype/chunks/codecs per lazy var).

    Only the lazy (``ManifestArray``) variables carry an encoding; eagerly-loaded
    coordinates are plain numpy and are intentionally excluded. Mixed NetCDF3/NetCDF4
    files of the same variable differ here (codec/chunking) and so partition apart.
    """
    sig = {}
    for name, var in vds.variables.items():
        meta = getattr(var.data, "metadata", None)
        if meta is None:
            continue  # loadable coordinate (real numpy) -> not part of the layout key
        sig[name] = [
            str(getattr(meta, "dtype", None)),
            list(getattr(meta, "chunks", []) or []),
            repr(getattr(meta, "codecs", getattr(meta, "compressor", None))),
        ]
    return json.dumps(sig, sort_keys=True)


def _partition(
    file_vds: Sequence[Tuple[str, xr.Dataset]]
) -> List[Tuple[str, List[Tuple[str, xr.Dataset]]]]:
    """Group ``(path, vds)`` pairs by identical encoding signature (first-seen order)."""
    groups: dict = {}
    order: List[str] = []
    for path, vds in file_vds:
        key = _signature(vds)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((path, vds))
    return [(k, groups[k]) for k in order]


# ------------------------------------------------------------- combine / open

def _combine(datasets: Sequence[xr.Dataset], combine: str, concat_dim) -> xr.Dataset:
    """Combine datasets the way the legacy direct open would.

    ``by_coords`` orders by coordinate values (robust to file naming and mixed
    NetCDF3/NetCDF4 formats); ``nested`` stacks in the given (input/file) order along
    ``concat_dim`` for bare index axes without a coordinate. A single dataset is
    returned unchanged. Used both to consolidate a partition at build time and to
    recombine the (few) partitions at open time.
    """
    datasets = list(datasets)
    if len(datasets) == 1:
        return datasets[0]
    if combine == COMBINE_NESTED:
        return xr.combine_nested(
            datasets, concat_dim=concat_dim, combine_attrs="override"
        )
    return xr.combine_by_coords(datasets, combine_attrs="override")


def _resolve_concat_dim(vds: xr.Dataset, concat_dim) -> Optional[str]:
    """The dimension to concatenate along: the explicit ``concat_dim`` if given, else
    the leading dimension of the largest data variable (the record/time axis — climate
    archives are time-first). No coordinate values are read (all-virtual)."""
    if concat_dim:
        return str(concat_dim)
    dvars = list(vds.data_vars)
    if not dvars:
        return None
    main = max(dvars, key=lambda n: vds[n].ndim)
    dims = vds[main].dims
    return str(dims[0]) if dims else None


def _consolidate(vds_list: Sequence[xr.Dataset], dim: Optional[str]) -> xr.Dataset:
    """Consolidate a signature-homogeneous partition into one virtual dataset (build time).

    A vectorized ``xr.concat`` along ``dim`` with ``data_vars="minimal"`` /
    ``coords="minimal"`` / ``compat="override"``: only variables that span ``dim`` are
    concatenated; everything else (auxiliary coordinates like 2-D lat/lon, the
    grid_mapping container) is taken from the first file unchanged — neither broadcast
    nor re-read for an equality check. Files are concatenated in input order; ``by_coords``
    ordering is applied lazily at open time (see :func:`open_store`), because a
    ``ManifestArray`` cannot be fancy-indexed (sortby) at build time.
    """
    vds_list = list(vds_list)
    if len(vds_list) == 1 or dim is None:
        return vds_list[0]
    return xr.concat(
        vds_list, dim=dim, data_vars="minimal", coords="minimal",
        compat="override", combine_attrs="override",
    )


def _regular_concat_grid(vds_list: Sequence[xr.Dataset], dim: Optional[str]) -> bool:
    """Whether the partition's files form a regular chunk grid for a build-time concat.

    VirtualiZarr only supports regular (evenly chunked) grids on the concatenation axis: a
    ``ManifestArray`` concat rejects any input *except the last* whose ``dim`` length is not
    an exact multiple of its ``dim`` chunk — a partial final chunk landing mid-array. This is
    the era5-hourly shape (yearly files of 8760 timesteps chunked at 512, ``8760 % 512 != 0``):
    every file carries a partial final chunk, so consolidating them into one manifest is
    impossible. When this returns ``False`` the caller keeps the files as per-file partitions
    and lets the reader concatenate the (dask-backed) datasets at open, which has no
    regular-grid constraint (see :func:`open_store`). A single file / no concat dim is trivially
    regular — nothing is concatenated at build time. Metadata is read the same way as
    :func:`_signature` (only lazy ``ManifestArray`` variables carry a chunk grid)."""
    vds_list = list(vds_list)
    if dim is None or len(vds_list) <= 1:
        return True
    for vds in vds_list[:-1]:  # the final input is allowed a partial final chunk
        if dim not in vds.sizes:
            continue
        length = int(vds.sizes[dim])
        for var in vds.variables.values():
            if dim not in var.dims:
                continue
            meta = getattr(var.data, "metadata", None)
            chunks = list(getattr(meta, "chunks", []) or []) if meta is not None else []
            if not chunks:
                continue  # unchunked/contiguous -> a single chunk spans the axis
            chunk = chunks[var.dims.index(dim)]
            if chunk and length % chunk:
                return False
    return True


def _record_coord(ds: xr.Dataset, dim: str) -> Optional[str]:
    """The 1-D coordinate that orders ``dim`` for a ``by_coords`` open.

    ``dim`` itself if it is a coordinate (the usual case: a dimension coordinate like
    ``time``), else the sole 1-D coordinate whose only dimension is ``dim`` — this is
    NSIDC's ``time`` over the *bare* record axis ``tdim`` (a dimension with no
    coordinate of its own). ``None`` if none or several candidates exist, in which case
    the open-time sort is skipped (there is nothing unambiguous to order by)."""
    if dim in ds.coords and ds[dim].ndim == 1:
        return dim
    cands = [c for c in ds.coords if ds[c].ndim == 1 and ds[c].dims == (dim,)]
    return cands[0] if len(cands) == 1 else None


def _record_values(ds: xr.Dataset, dim: Optional[str]) -> Optional[np.ndarray]:
    """The 1-D record-coordinate values ordering ``dim`` (read from the inline
    coordinate, never the data), or ``None`` if there is no unambiguous coordinate."""
    if dim is None:
        return None
    rc = _record_coord(ds, dim)
    if rc is None:
        return None
    return np.asarray(ds[rc].values).ravel()


def _record_start(ds: xr.Dataset, dim: Optional[str]):
    """Smallest value of the record coordinate ordering ``dim``, or ``None``.

    Used at build time to order partition groups by record so the reader concatenates
    them chronologically and skips the open-time ``sortby``."""
    vals = _record_values(ds, dim)
    return vals.min() if vals is not None and vals.size else None


def _axis_key(ds: xr.Dataset, dim: Optional[str]) -> Optional[str]:
    """A stable hash of the record-coordinate *values*, identifying partitions that
    share the **same record axis** (different variables of one cube, each in its own
    single-variable file -> its own encoding partition). Such partitions must be
    *merged* at open (not concatenated). ``None`` if there is no orderable coordinate."""
    vals = _record_values(ds, dim)
    if vals is None or not vals.size:
        return None
    return hashlib.sha1(np.ascontiguousarray(vals).tobytes()).hexdigest()


def open_reference(ref, *, open_kwargs: Optional[Mapping] = None) -> xr.Dataset:
    """Open a manifest as a lazy, CF-decoded xarray Dataset.

    ``ref`` is a path to a parquet manifest directory (consolidated partition) or to a
    kerchunk JSON file (per-file manifest). Parquet manifests are loaded lazily via
    pyarrow, so only array metadata + inline coordinates are read here — never data.

    ``open_kwargs`` (xarray opener options: ``chunks`` and decode flags) override the
    hardcoded defaults ``{decode_coords: "all", chunks: {}}``. ``chunks`` is the lever
    that bounds the dask task graph: ``chunks={}`` mirrors the on-disk chunking (one task
    per chunk, O(#chunks) at open), whereas e.g. ``chunks={"time": 1000}`` coalesces the
    record axis into far fewer blocks while keeping reads lazy.
    """
    opts = {**_OPEN_DEFAULTS, **_filter_open_kwargs(open_kwargs)}
    is_parquet = isinstance(ref, str) and os.path.isdir(ref)
    # Lazy parquet refs are auto-detected from the directory layout (LazyReferenceMapper);
    # fsspec has no ``lazy=`` kwarg here (it would be silently ignored), so we don't pass one.
    fs = fsspec.filesystem("reference", fo=ref, remote_protocol="file")
    return xr.open_dataset(
        fs.get_mapper(""), engine="zarr", consolidated=False, **opts,
    )


# --------------------------------------------------------------- store payloads

def _ref_file(cache_dir: str, posix_path: str) -> str:
    h = hashlib.sha1(posix_path.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, FILES_SUBDIR, h + ".json")


def _prune_ref_files(cache_dir: str, keep_posix: set) -> None:
    """Remove per-file manifest JSONs no longer in the current file set."""
    keep = {os.path.basename(_ref_file(cache_dir, p)) for p in keep_posix}
    files_dir = os.path.join(cache_dir, FILES_SUBDIR)
    if not os.path.isdir(files_dir):
        return
    for name in os.listdir(files_dir):
        if name not in keep:
            try:
                os.remove(os.path.join(files_dir, name))
            except FileNotFoundError:
                pass


# The scalar sidecar is the only part of ``store.json`` that carries arbitrary
# (non-string) values, and a CF grid-mapping container can be a byte/char scalar
# (e.g. NSIDC ``projection`` is ``|S1`` with value ``b''``) or carry byte-valued
# attrs. ``json`` cannot serialize ``bytes``, so we tag them with a reversible
# base64 wrapper on the way out and restore them on the way in.
_BYTES_TAG = "__geokube_bytes_b64__"


def _encode_bytes(obj):
    """Recursively replace ``bytes`` with a JSON-safe, reversible base64 tag."""
    if isinstance(obj, (bytes, bytearray)):
        return {_BYTES_TAG: base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _encode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_bytes(v) for v in obj]
    return obj


def _decode_bytes(obj):
    """Inverse of :func:`_encode_bytes`."""
    if isinstance(obj, dict):
        if len(obj) == 1 and _BYTES_TAG in obj:
            return base64.b64decode(obj[_BYTES_TAG])
        return {k: _decode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bytes(v) for v in obj]
    return obj


def _extract_scalars(
    ds: xr.Dataset, src_file: str, *, open_kwargs: Optional[Mapping] = None
) -> Tuple[xr.Dataset, dict, dict]:
    """Pull 0-dim variables out of ``ds`` into a JSON sidecar, reading their values
    directly from ``src_file``; also capture each data var's grid_mapping link.

    VirtualiZarr cannot round-trip scalar variables through a kerchunk manifest (a
    virtual 0-dim ``ManifestArray`` is unreadable, and an eagerly-loaded one serializes
    to a fill value). Scalars are tiny and identical across a partition's files, so we
    read them once from the first file, persist ``{value, attrs, dtype, is_coord}`` in
    the store, drop them from the manifest, and re-attach them at open. This covers the
    common scalar reference level (e.g. ``height_2m``) and the grid_mapping container
    (e.g. ``rotated_pole``) that carries the CRS. Dropping the grid_mapping container
    breaks the data var's ``grid_mapping`` link during decode, so we record it
    (``{data_var: grid_mapping_var}``) and restore it at open.
    """
    sidecar, gridmap = {}, {}
    decode = {"decode_coords": "all", **_decode_open_kwargs(open_kwargs)}
    with xr.open_dataset(src_file, **decode) as src:
        names = [str(v) for v in src.variables if src[v].ndim == 0]
        for v in src.data_vars:
            gm = src[v].encoding.get("grid_mapping") or src[v].attrs.get("grid_mapping")
            if gm:
                gridmap[str(v)] = str(gm)
        for n in names:
            sidecar[n] = {
                "data": np.asarray(src[n].values).item(),
                "attrs": dict(src[n].attrs),
                "dtype": str(src[n].dtype),
                # CF-decoded source marks scalar coords / the grid_mapping container as
                # coordinates; mirror that so the reopened cube matches a direct open.
                "is_coord": n in src.coords,
            }
    drop = [n for n in names if n in ds.variables]
    return (ds.drop_vars(drop) if drop else ds), _encode_bytes(sidecar), gridmap


def _reattach_scalars(ds: xr.Dataset, sidecar: Mapping, gridmap: Mapping) -> xr.Dataset:
    """Inverse of :func:`_extract_scalars`: restore the 0-dim variables + grid_mapping."""
    sidecar = _decode_bytes(dict(sidecar or {}))
    for name, info in sidecar.items():
        ds[name] = xr.DataArray(
            np.array(info["data"], dtype=info["dtype"]), attrs=dict(info.get("attrs", {}))
        )
        if info.get("is_coord"):
            ds = ds.set_coords([name])
    for dvar, gm in (gridmap or {}).items():
        if dvar in ds.variables and gm in ds.variables:
            ds[dvar].encoding["grid_mapping"] = gm
    return ds


def _write_manifest(ds: xr.Dataset, abs_path: str) -> None:
    """(Over)write a consolidated manifest at ``abs_path`` in ``_COMBINED_FORMAT``."""
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path, ignore_errors=True)
    elif os.path.isfile(abs_path):
        os.remove(abs_path)
    _to_kerchunk(ds, abs_path, _COMBINED_FORMAT)


def _inline_coords(
    combined: xr.Dataset, paths: Sequence[str], dim: Optional[str],
    *, open_kwargs: Optional[Mapping] = None, progress: bool = False,
) -> xr.Dataset:
    """Materialize the consolidated dataset's non-scalar coordinates as inline numpy.

    The concat coordinate is rebuilt by concatenating each file's CF-*decoded* values
    (decoded with that file's own ``units``), so per-file unit rebasing — every
    single-timestep file encoded as ``since <its own date>`` with raw value 0 — does
    NOT collide (the trap the old ``cf:`` MZZ selector handled; a raw manifest concat
    would otherwise yield one timestep repeated). Other non-scalar coordinates are
    identical across the partition and taken from the first file. Inlining keeps opens
    instant (no per-record coordinate reads). Reads are small 1-D coordinate slices, not
    the data, so the incremental win (skipping the per-file manifest build) holds.
    """
    decode = {"decode_coords": "all", **_decode_open_kwargs(open_kwargs), "chunks": {}}
    # First file gives the coordinate set, dims, attrs, and the non-spanning values.
    with xr.open_dataset(paths[0], **decode) as s0:
        spanning, nonspan = {}, {}
        for c in s0.coords:
            cn = str(c)
            if s0[c].ndim == 0:
                continue  # scalar -> handled by the sidecar
            meta = (s0[c].dims, dict(s0[c].attrs))
            if dim is not None and dim in s0[c].dims:
                spanning[cn] = meta
            else:
                nonspan[cn] = (s0[c].dims, np.asarray(s0[c].values), dict(s0[c].attrs))

    coord_vals = dict(nonspan)
    if spanning:
        # Coordinates that span the concat dim (the time axis and its bounds) are rebuilt
        # by concatenating each file's CF-decoded values — one open per file, decoded with
        # that file's own units, so per-file unit rebasing does not collide.
        seg = {cn: [] for cn in spanning}
        for pth in progress_iter(
            paths, enabled=progress, desc="inlining coords", total=len(paths),
            leave=False,
        ):
            with xr.open_dataset(pth, **decode) as s:
                for cn in spanning:
                    seg[cn].append(np.asarray(s[cn].values))
        for cn, (dims, attrs) in spanning.items():
            axis = dims.index(dim)
            coord_vals[cn] = (dims, np.concatenate(seg[cn], axis=axis), attrs)
    return combined.assign_coords(coord_vals) if coord_vals else combined


def _combine_plan(
    metas: Sequence[Tuple[object, Optional[str]]]
) -> Optional[List[List[int]]]:
    """Build the recombination plan from per-partition ``(record_start, axis_key)``.

    Partitions sharing an ``axis_key`` (identical record coordinate) are different
    variables over the same axis -> one **merge group**. Groups are ordered by their
    record start, giving the concat order; the reader merges within a group and
    concatenates across groups. Returns ``None`` (reader falls back to the plain concat
    path) for a single partition or when any partition lacks a start/axis_key (no
    unambiguous record coordinate — e.g. a bare index axis)."""
    if len(metas) <= 1 or any(s is None or a is None for s, a in metas):
        return None
    groups: dict = {}
    start_of: dict = {}
    order: List[str] = []
    for idx, (start, axis) in enumerate(metas):
        if axis not in groups:
            groups[axis] = []
            start_of[axis] = start
            order.append(axis)
        groups[axis].append(idx)
    order.sort(key=lambda a: start_of[a])  # concat order: groups by record start
    return [groups[a] for a in order]


def _assemble_store(
    file_vds: Sequence[Tuple[str, xr.Dataset]],
    cache_dir: str,
    *,
    combine: str,
    concat_dim,
    open_kwargs: Optional[Mapping] = None,
    progress: bool = False,
    passthrough_contiguous: bool = True,
    passthrough_max_files_per_partition: int = DEFAULT_PASSTHROUGH_MAX_FILES,
) -> dict:
    """Partition the per-file virtual datasets, consolidate each, write a manifest + index.

    Each encoding-homogeneous partition is consolidated via :func:`_consolidate` (a
    vectorized manifest concat) — unless its files would form an *irregular* chunk grid on
    the concat axis (:func:`_regular_concat_grid`; the era5-hourly shape of 8760 timesteps
    chunked at 512). VirtualiZarr forbids that concat, so such a group is instead emitted as
    **one partition per file** and recombined at open by the dask-backed reader (which has no
    regular-grid constraint) via the same ``combine_plan`` path as disjoint tiles. The
    consolidated non-scalar coordinates are then
    *materialized inline* (read once via an :func:`open_reference` round-trip and
    re-assigned as numpy) so the persisted manifest carries its coordinates as a single
    inline blob — the reader never does per-record coordinate disk reads, keeping opens
    fast. Scalars are sidecar'd (:func:`_extract_scalars`). ``store.json`` records the
    partition index (relative manifest paths, file list, signature, scalar sidecar) plus
    the combine spec and the resolved concat dim (used for the lazy open-time sort).

    For ``by_coords`` it also persists a ``combine_plan`` (:func:`_combine_plan`): partitions
    are grouped by **record-axis identity** (:func:`_axis_key`) — partitions sharing the
    same record coordinate are different variables of one cube (each single-variable file is
    its own encoding partition) and must be *merged* at open; groups with distinct axes are
    *concatenated* along the record dim, ordered by start so the reader skips ``sortby`` for
    disjoint ranges (the ``sortby`` in :func:`open_store` stays the fallback for interleaved
    ranges). This replays ``open_mfdataset(by_coords)`` semantics (merge + concat) instead of
    a blind ``xr.concat`` that would stack per-variable partitions and inflate the record axis.

    ``passthrough_contiguous`` (default ``True``) additionally detects, per encoding-signature
    group, data variables whose native chunk spans the whole array (:func:`_detect_passthrough`)
    -- kerchunk can only reference such a variable as one indivisible chunk, so no ``chunks=``
    at read time can subdivide it. These are dropped from the manifest entirely (no reference
    is ever written for them) and recorded instead as a lightweight ``passthrough`` descriptor
    (source file paths + native engine) on the partition entry, reopened directly from source
    at read time by :func:`_splice_passthrough`. A variable spanning more than
    ``passthrough_max_files_per_partition`` files within one partition falls back to today's
    plain single-chunk reference instead (with a warning), to keep the read-time cost of
    reopening files natively bounded.
    """
    parts_dir = os.path.join(cache_dir, PARTS_SUBDIR)
    if os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)

    ext = ".json" if _COMBINED_FORMAT == "json" else ""
    resolved_dim = str(concat_dim) if concat_dim else None
    partitions: List[dict] = []
    # Per-partition (record-start, axis-key) in natural (first-seen) order; feeds the
    # combine plan below. The partition list itself is NOT reordered — the plan refers
    # to partitions by index and carries both the merge grouping and the concat order.
    metas: List[Tuple[object, Optional[str]]] = []
    for sig_key, items in _partition(file_vds):
        vds_list = [v for _, v in items]
        paths = [p for p, _ in items]
        dim = _resolve_concat_dim(vds_list[0], concat_dim)
        resolved_dim = resolved_dim or dim

        # Detected once per encoding-signature group: `_signature` folds the native chunk
        # shape into the grouping key, so every file in `vds_list` shares an identical
        # per-file chunk/shape for a given variable -- checking one file stands for the
        # whole group, regardless of how the subgroups below get consolidated.
        passthrough_vars = _detect_passthrough(vds_list[0], enabled=passthrough_contiguous)
        _warn_if_many_small_chunks(vds_list[0], dim, len(vds_list), open_kwargs, passthrough_vars)

        # Normally the whole (encoding-homogeneous) group is consolidated into one manifest
        # via a vectorized ManifestArray concat. VirtualiZarr only supports regular chunk
        # grids on the concat axis, though: when the files' ``dim`` length is not an exact
        # multiple of their ``dim`` chunk (the era5-hourly shape: 8760 timesteps chunked at
        # 512) the concat is rejected, so fall back to per-file partitions — the reader then
        # concatenates the dask-backed datasets at open, which has no regular-grid constraint.
        # ``subgroups`` is a list of (dataset, paths): one consolidated dataset, or one entry
        # per file. The try/except is a backstop for irregular grids the pre-check misses.
        if _regular_concat_grid(vds_list, dim):
            try:
                subgroups = [(_consolidate(vds_list, dim), paths)]
            except ValueError as exc:
                if "partial chunks" not in str(exc):
                    raise
                subgroups = [(v, [p]) for v, p in zip(vds_list, paths)]
        else:
            subgroups = [(v, [p]) for v, p in zip(vds_list, paths)]

        for ds, ds_paths in subgroups:
            group_passthrough = {}
            if passthrough_vars:
                if len(ds_paths) > passthrough_max_files_per_partition:
                    LOG.warn(
                        f"variables {sorted(passthrough_vars)} are contiguous/unchunked at"
                        f" the source but span {len(ds_paths)} files in one partition"
                        f" (cap {passthrough_max_files_per_partition}); falling back to a"
                        " single-chunk kerchunk reference for them instead of reopening"
                        " that many files natively at read time."
                    )
                else:
                    group_passthrough = passthrough_vars

            combined = _inline_coords(
                ds, ds_paths, dim, open_kwargs=open_kwargs, progress=progress
            )
            start, axis = _record_start(combined, dim), _axis_key(combined, dim)
            combined, scalars, gridmap = _extract_scalars(
                combined, ds_paths[0], open_kwargs=open_kwargs
            )
            if group_passthrough:
                # No kerchunk reference at all for these -- reopened directly from
                # `ds_paths` at read time (`_splice_passthrough`). Dropping them here keeps
                # a single source of truth: never both a reference AND a passthrough splice.
                combined = combined.drop_vars(list(group_passthrough), errors="ignore")

            rel = os.path.join(PARTS_SUBDIR, f"p{len(partitions):04d}{ext}")
            _write_manifest(combined, os.path.join(cache_dir, rel))

            entry = {
                "signature": sig_key, "files": ds_paths, "parquet": rel,
                "scalars": scalars, "gridmap": gridmap,
            }
            if group_passthrough:
                fmt = detect_format(ds_paths[0])
                entry["passthrough"] = {
                    name: {"files": ds_paths, "engine": _PASSTHROUGH_ENGINE.get(fmt, fmt)}
                    for name in group_passthrough
                }
            partitions.append(entry)
            metas.append((start, axis))

    plan = _combine_plan(metas) if combine == COMBINE_BY_COORDS else None

    payload = {
        "vz_version": VZ_VERSION,
        "store_schema": STORE_SCHEMA_VERSION,
        "combine": combine,
        "concat_dim": resolved_dim if combine == COMBINE_BY_COORDS else concat_dim,
        # Persisted opener kwargs replayed by the reader (overridable at read time).
        # Additive field: stores written before this read as {} -> reader uses defaults.
        "open_kwargs": _filter_open_kwargs(open_kwargs),
        "partitions": partitions,
        # Merge-aware recombination plan (None -> reader falls back to the plain
        # concat path; absent in pre-schema-4 stores, which rebuild anyway).
        "combine_plan": plan,
    }
    _cache.write_json(os.path.join(cache_dir, STORE_FILE), payload)
    return payload


def _run_parallel(thunks: Sequence, scheduler, *, progress: bool = False, desc=None):
    """Run zero-arg callables, optionally via dask (returning results in order).

    ``None`` / inactive ``"auto"`` -> serial in-process. Otherwise the thunks are
    wrapped in ``dask.delayed`` and forwarded to :func:`dask.compute` with the given
    scheduler. ``"threads"`` is the recommended parallel mode: the per-file open is
    I/O-bound (file-structure read) and the returned virtual datasets stay in shared
    memory (no pickling), which matters for slow/network storage.

    With ``progress`` a tqdm bar (labelled ``desc``) tracks per-file completion: on the
    serial paths it wraps the loop; on the dask path a ``dask.diagnostics`` callback drives
    it (local schedulers only). Under a distributed scheduler that callback never fires, so
    a single log line is emitted instead.
    """
    if not thunks:
        return []
    if scheduler is None:
        return [
            t() for t in progress_iter(
                thunks, enabled=progress, desc=desc, total=len(thunks), leave=False
            )
        ]
    import dask
    from dask.base import get_scheduler

    if scheduler == SCHEDULER_AUTO and get_scheduler() is None:
        return [
            t() for t in progress_iter(
                thunks, enabled=progress, desc=desc, total=len(thunks), leave=False
            )
        ]
    tasks = [dask.delayed(t)() for t in thunks]
    compute_kwargs = {} if scheduler == SCHEDULER_AUTO else {"scheduler": scheduler}
    if progress and is_distributed(scheduler):
        LOG.info(
            f"{desc or 'processing'}: {len(tasks)} files"
            " (distributed scheduler; per-file bar unavailable)"
        )
        return list(dask.compute(*tasks, **compute_kwargs))
    with dask_progress(len(tasks), enabled=progress, desc=desc, leave=False):
        return list(dask.compute(*tasks, **compute_kwargs))


def cached_build_store(
    files: Sequence[str],
    cache_dir: str,
    *,
    reuse_keys: Sequence[str] = (),
    combine: str = COMBINE_BY_COORDS,
    concat_dim=None,
    scheduler=SCHEDULER_AUTO,
    open_kwargs: Optional[Mapping] = None,
    progress: bool = False,
    passthrough_contiguous: bool = True,
    passthrough_max_files_per_partition: int = DEFAULT_PASSTHROUGH_MAX_FILES,
) -> Optional[dict]:
    """Incrementally (re)build the on-disk store under ``cache_dir``.

    Files whose POSIX path is in ``reuse_keys`` are re-hydrated from their cached
    per-file JSON manifest; the rest (the *stale* set) are opened fresh via
    :func:`reference_one` — the parallelizable step (honours ``scheduler``). Stale
    manifests are persisted, removed ones pruned, the partitions are consolidated to
    parquet and ``store.json`` is written. Returns the payload, or ``None`` (no writes)
    if a file is not referenceable — the caller then falls back to ``open_mfdataset``.

    With ``progress`` a per-file tqdm bar tracks the opens (fresh + reused) and the
    coordinate-inlining pass.
    """
    if not all(is_referenceable(f) for f in files):
        return None
    open_kwargs = dict(open_kwargs or {})
    # Robustness: if the caller did not pin ``decode_times`` and any file carries a non-CF
    # ``months``/``years since`` unit (undecodable by xarray/cftime for a real-world
    # calendar), force ``decode_times=False`` for the whole build so the manifest read never
    # crashes. It flows to every build opener and is persisted in the store (``_assemble_store``)
    # so the reader replays it -- geokube decodes such time to datetime64 at read. ``any(...)``
    # short-circuits on the first undecodable file.
    if "decode_times" not in open_kwargs and any(
        _time_units_undecodable(f) for f in files
    ):
        open_kwargs["decode_times"] = False
    os.makedirs(os.path.join(cache_dir, FILES_SUBDIR), exist_ok=True)
    reuse = set(reuse_keys)
    posix_paths = [_make_path_posix(f) for f in files]
    stale_idx = [i for i, p in enumerate(posix_paths) if p not in reuse]
    reuse_idx = [i for i, p in enumerate(posix_paths) if p in reuse]

    # Stale files: open fresh (parallel) and persist each per-file JSON manifest.
    fresh = _run_parallel(
        [(lambda f=files[i]: reference_one(f, open_kwargs=open_kwargs)) for i in stale_idx],
        scheduler,
        progress=progress,
        desc="opening files",
    )
    vds_by_idx = {}
    for k, i in enumerate(stale_idx):
        vds = fresh[k]
        _to_kerchunk(vds, _ref_file(cache_dir, posix_paths[i]), "json")
        vds_by_idx[i] = vds

    # Reused files: re-hydrate from the cached JSON (parallel); regenerate on a miss.
    reloaded = _run_parallel(
        [(lambda i=i: _reload_one(_ref_file(cache_dir, posix_paths[i]), open_kwargs=open_kwargs))
         for i in reuse_idx],
        scheduler,
        progress=progress,
        desc="reloading cached",
    )
    for k, i in enumerate(reuse_idx):
        vds = reloaded[k]
        if vds is None:  # rare cache miss -> regenerate and persist
            vds = reference_one(files[i], open_kwargs=open_kwargs)
            _to_kerchunk(vds, _ref_file(cache_dir, posix_paths[i]), "json")
        vds_by_idx[i] = vds

    file_vds = [(files[i], vds_by_idx[i]) for i in range(len(files))]
    payload = _assemble_store(
        file_vds, cache_dir, combine=combine, concat_dim=concat_dim,
        open_kwargs=open_kwargs, progress=progress,
        passthrough_contiguous=passthrough_contiguous,
        passthrough_max_files_per_partition=passthrough_max_files_per_partition,
    )
    _prune_ref_files(cache_dir, set(posix_paths))
    return payload


def load_store(cache_dir: str) -> Optional[dict]:
    """Load the store index from ``cache_dir`` (or ``None``).

    The ``cache_dir`` is injected into the payload so :func:`open_store` can resolve
    the partitions' relative parquet paths (kept relative on disk for portability).
    """
    payload = _cache.read_json(os.path.join(cache_dir, STORE_FILE))
    if payload is not None:
        payload["_cache_dir"] = cache_dir
    return payload


def _is_monotonic_nondecreasing(out: xr.Dataset, rc: str) -> bool:
    """True if the 1-D record coordinate ``rc`` is already sorted non-decreasing.

    Uses the eagerly-materialized pandas index when ``rc`` is a dimension coordinate;
    otherwise reads the (small, inline) 1-D coordinate values. Never touches data."""
    if rc in out.indexes:
        return bool(out.indexes[rc].is_monotonic_increasing)
    vals = np.asarray(out[rc].values).ravel()
    return vals.size <= 1 or bool(np.all(vals[1:] >= vals[:-1]))


def _merge_partitions(group: Sequence[xr.Dataset]) -> xr.Dataset:
    """Merge partitions that share a record axis (different variables of one cube, each
    in its own single-variable file -> its own encoding partition).

    ``join="override"`` / ``compat="override"`` skip alignment and equality checks: the
    record axes are identical by construction (same :func:`_axis_key`), so this just
    unions the data variables and keeps the first copy of the shared coordinates /
    grid_mapping containers. Variables stay lazy."""
    group = list(group)
    if len(group) == 1:
        return group[0]
    return xr.merge(
        group, compat="override", join="override", combine_attrs="override"
    )


def _passthrough_chunks(open_kwargs: Optional[Mapping]):
    """The ``chunks=`` value to reopen a passthrough variable with.

    The merged ``chunks`` (persisted default overridden at read time) governs the
    *reference* path, where the bare ``{}`` default deliberately mirrors on-disk
    chunking. A contiguous variable opened via its native engine has no on-disk chunk
    to mirror at all -- h5netcdf/netCDF4 never set ``encoding["preferred_chunks"]``
    for it -- and xarray's own chunk resolution turns a bare ``{}``/unset ``chunks``
    into a *single whole-array* chunk whenever no preferred chunk is exposed, silently
    recreating the very problem passthrough exists to avoid. ``"auto"`` has no such
    trap: with no preferred chunk to align to, dask sizes the chunk from the byte-size
    target alone -- exactly the "on the fly" chunking a normal (non-cached) open
    already gets. Only an explicit, non-empty ``chunks`` dict overrides this default.
    """
    chunks = (open_kwargs or {}).get("chunks")
    return chunks if chunks else "auto"


def _splice_passthrough(
    ds: xr.Dataset, entry: Mapping, dim: Optional[str], open_kwargs: Optional[Mapping],
) -> xr.Dataset:
    """Reopen "passthrough" variables (:func:`_detect_passthrough`) directly from
    their original file(s), splicing them into ``ds`` in place of what would
    otherwise be one indivisible kerchunk-reference chunk. Runs before any
    cross-partition merge/concat/sortby in :func:`open_store`, so a spliced variable
    rides through that logic identically to a reference-derived one -- no separate
    combine path is needed.

    Only the reopened variable's own values/dims/attrs/encoding are kept -- its own
    coordinates are discarded so ``ds`` keeps exactly one coordinate object per
    dimension (the one already inlined from the reference path).

    ``drop_variables`` is intentionally NOT forwarded to the *build* (see
    ``geolake-datastore``'s ``intake_geokube.base._BUILD_ONLY_XARRAY_KWARGS``): a
    variable excluded at read can still be a passthrough candidate at build (e.g.
    bioclimind's single-record ``time_bnds``, contiguous but absent from only some
    of the dataset's files). The reference path already drops such a name for free
    via xarray's own lenient ``drop_variables``; a passthrough name in the current
    ``drop_variables`` is skipped here the same way -- it was already excluded from
    the manifest at build time (:func:`_assemble_store`), so skipping the splice
    just leaves it absent from the result, instead of reopening a file and indexing
    a variable that open already dropped.
    """
    passthrough = entry.get("passthrough") or {}
    if not passthrough:
        return ds
    drop = set((open_kwargs or {}).get("drop_variables") or ())
    passthrough = {k: v for k, v in passthrough.items() if k not in drop}
    if not passthrough:
        return ds
    decode = {"decode_coords": "all", **_decode_open_kwargs(open_kwargs)}
    chunks = _passthrough_chunks(open_kwargs)
    for name, info in passthrough.items():
        files, engine = info["files"], info["engine"]
        opened_full = [
            xr.open_dataset(f, engine=engine, chunks=chunks, **decode) for f in files
        ]
        # `decode_coords="all"` promotes CF bounds/grid_mapping variables (e.g.
        # `time_bnds`) to coordinates on a direct open; mirror that status so the
        # spliced variable matches a non-cached open's `data_vars`/`coords` split.
        is_coord = name in opened_full[0].coords
        opened = [o[name] for o in opened_full]
        resolved = dim or (opened[0].dims[0] if opened[0].dims else None)
        if len(opened) == 1:
            var = opened[0]
        elif resolved and resolved in opened[0].dims:
            # `data_vars`/`coords` (used elsewhere for a Dataset concat) are not valid
            # for a plain DataArray concat -- there is nothing else to minimize here.
            var = xr.concat(
                opened, dim=resolved, join="override", combine_attrs="override",
            )
        else:
            var = opened[0]
        ds[name] = xr.Variable(var.dims, var.data, dict(var.attrs), dict(var.encoding))
        if is_coord:
            ds = ds.set_coords([name])
    return ds


def open_store(payload: Mapping, *, open_kwargs: Optional[Mapping] = None) -> xr.Dataset:
    """Reopen the combined lazy dataset from a store payload.

    The ``combine_plan`` (schema 4+) drives a merge-aware recombination that replays
    ``open_mfdataset(by_coords)`` semantics: partitions sharing a record axis (different
    variables of one cube) are **merged** (:func:`_merge_partitions`), and the resulting
    distinct-axis groups are **concatenated** along the record dim in record order. This
    avoids stacking per-variable partitions along the record axis (which would inflate it
    N-fold and produce a wrong cube). With one group the merge is the whole result; with
    one partition it is a single lazy reference open. ``nested`` (and pre-schema-4 / planless
    ``by_coords``) multi-partition stores use a plain **lazy** ``xr.concat`` along the concat
    dim in partition (input/file) order — never an eager ``combine_nested``/``combine_by_coords``
    (whose default ``compat`` would compute variables and OOM). Only the inline coordinates are
    touched (merged/concatenated/argsorted) — the data variables stay lazy.

    ``open_kwargs`` (xarray opener options) override the kwargs persisted at build time;
    both are forwarded to :func:`open_reference`. ``chunks`` is the lever that keeps the
    dask graph bounded at scale.
    """
    cache_dir = payload.get("_cache_dir", "")
    parts = payload["partitions"]
    combine = payload.get("combine", COMBINE_BY_COORDS)
    concat_dim = payload.get("concat_dim")
    plan = payload.get("combine_plan")
    # Persisted build-time kwargs are the defaults; read-time kwargs win.
    merged_open_kwargs = {**payload.get("open_kwargs", {}), **(open_kwargs or {})}
    datasets = [
        _splice_passthrough(
            _reattach_scalars(
                open_reference(
                    os.path.join(cache_dir, p["parquet"]), open_kwargs=merged_open_kwargs
                ),
                p.get("scalars", {}), p.get("gridmap", {}),
            ),
            p, concat_dim, merged_open_kwargs,
        )
        for p in parts
    ]
    if not concat_dim and len(datasets) > 1:
        # Some stores persist no concat dim -- notably ``nested`` built without an explicit
        # one (``:692`` keeps the raw ``None``). Resolve it from the opened data (leading dim
        # of the largest var, the record axis) so the lazy ``xr.concat`` branch below applies
        # instead of the eager ``_combine`` (whose default ``compat`` computes a variable
        # aligned across the partitions' disjoint ranges -> OOM). No cache rebuild needed.
        concat_dim = _resolve_concat_dim(datasets[0], None)
    if combine == COMBINE_BY_COORDS and plan:
        # Merge-aware: merge partitions within each record-axis group, then concatenate
        # the (distinct-axis) groups along the record dim. Groups are ordered by record
        # start at build, so the concat is already chronological for disjoint ranges and
        # the sortby below is skipped; the sortby stays the fallback for interleaved ranges.
        groups = [_merge_partitions([datasets[i] for i in g]) for g in plan]
        if len(groups) == 1:
            out = groups[0]
        elif concat_dim:
            out = xr.concat(
                groups, dim=concat_dim, data_vars="minimal", coords="minimal",
                compat="override", combine_attrs="override",
            )
        else:
            return _combine(groups, combine, concat_dim)
    elif len(datasets) == 1:
        out = datasets[0]
    elif concat_dim:
        # Multi-partition with a resolved concat dim: plain LAZY concat along it in partition
        # (== input/file) order. ``minimal``/``override`` keep it lazy — no per-variable
        # equality compute (the OOM trap of ``xr.combine_nested``'s / ``combine_by_coords``'s
        # default ``compat``, seen when a per-file split yields many disjoint-range partitions).
        # ``by_coords`` (no plan: pre-schema-4 / no orderable record coord) is sorted below and
        # is robust to interleaved ranges; ``nested`` keeps input order (the sort is skipped).
        out = xr.concat(
            datasets, dim=concat_dim, data_vars="minimal", coords="minimal",
            compat="override", combine_attrs="override",
        )
    else:
        # No resolved concat dim (degenerate): replay the recorded combine.
        return _combine(datasets, combine, concat_dim)
    # by_coords: order by the concat coordinate. The sort is deferred to here (a build-time
    # ManifestArray cannot be fancy-indexed) and runs on the dask-backed open, so it only
    # reorders the lazy graph — never reads data. Covers the single-group/single-partition
    # result and the concatenated multi-group one above. For a bare record axis (no coordinate
    # of its own, e.g. NSIDC ``tdim``) we order by the coordinate that spans it (``time``);
    # ``sortby(concat_dim)`` would otherwise raise on the missing ``tdim`` coordinate.
    if combine == COMBINE_BY_COORDS and concat_dim and concat_dim in out.dims:
        rc = _record_coord(out, concat_dim)
        # Skip the sort when the coordinate is already non-decreasing (the common case:
        # one partition, files in chronological order). ``sortby`` always builds an
        # ``np.lexsort`` + fancy-index ``isel`` graph — O(#chunks) along the record axis,
        # which at ~hundreds of thousands of chunks is a needless multi-GB graph. The
        # check runs on the small 1-D coordinate only; data stays lazy either way.
        if rc is not None and not _is_monotonic_nondecreasing(out, rc):
            out = out.sortby(rc)
    return out
