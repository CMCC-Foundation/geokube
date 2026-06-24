"""Kerchunk reference payloads for lazy multi-file ``DataCube`` caching.

Opening hundreds/thousands of NetCDF files with ``xr.open_mfdataset`` can take
hours: the cost is the *open* (per-file metadata read + coordinate alignment), not
the data processing. This module persists that per-file metadata as **kerchunk
reference JSON** instead of a pickle, so subsequent opens are cheap and the cache is
portable, secure and robust across library versions.

Design:

* Each file gets a per-file reference via the maker matching its on-disk format
  (HDF5/NetCDF4 -> :class:`SingleHdf5ToZarr`; NetCDF3-classic ->
  :class:`NetCDF3ToZarr`), chosen by magic-byte sniffing. These are persisted under
  ``files/`` for incremental rebuilds.
* The combined store (:data:`STORE_FILE`) holds the list of per-file references plus
  a **combine spec**. At open time the references are opened lazily and recombined
  with xarray exactly as ``open_mfdataset`` would — ``combine_by_coords`` (the
  default, infers ordering from coordinates and so handles files named out of
  coordinate order *and* mixed NetCDF3/NetCDF4 formats), or ``combine_nested`` along
  an explicit ``concat_dim`` for archives whose concat axis is a bare *index*
  dimension with no coordinate (e.g. NSIDC ``tdim``, whose timestamps live in a
  separate ``time(tdim)`` aux coordinate).

Recombining at the decoded-array level (rather than pre-concatenating references)
keeps the combine identical to the legacy direct open, while still paying the
per-file metadata read only once. The payload is produced by :func:`build_store` /
:func:`cached_build_store` and reopened by :func:`open_store`.
"""
from __future__ import annotations

__all__ = [
    "KERCHUNK_VERSION",
    "detect_format",
    "is_referenceable",
    "reference_one",
    "open_reference",
    "build_store",
    "cached_build_store",
    "load_store",
    "open_store",
]

import hashlib
import importlib
import os
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import fsspec
import xarray as xr

import kerchunk
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.netCDF3 import NetCDF3ToZarr

from geokube.backend import _cache
from geokube.utils.format_parsing import _make_path_posix

KERCHUNK_VERSION = kerchunk.__version__

# Cache-directory layout for the per-cube kerchunk store.
FILES_SUBDIR = "files"      # one per-file reference JSON (enables incremental rebuild)
STORE_FILE = "store.json"   # per-file references + combine spec

# Combine strategies persisted in the store and replayed at open time.
COMBINE_BY_COORDS = "by_coords"
COMBINE_NESTED = "nested"


def _patch_zarr_fill_value() -> None:
    """Compatibility shim for numpy>=2 + zarr 2.18 + kerchunk 0.2.7.

    ``zarr.meta.encode_fill_value`` does ``int(v)`` / ``float(v)``, which raises on
    a 1-element-array ``_FillValue`` because numpy 2 forbids implicit array->scalar
    conversion. kerchunk catches that error and silently DROPS the variable — a
    data-loss trap that hits packed int16 climate data (a very common layout). We
    wrap ``encode_fill_value`` to squeeze a size-1 array to a 0-d scalar first.
    Idempotent; applied only when this (cache-only) module is imported.
    """
    import zarr.meta as zmeta

    orig = zmeta.encode_fill_value
    if getattr(orig, "_geokube_squeeze", False):
        return

    def encode_fill_value(v, dtype, object_codec=None):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.size == 1:
            v = v.reshape(())[()]
        return orig(v, dtype, object_codec)

    encode_fill_value._geokube_squeeze = True
    zmeta.encode_fill_value = encode_fill_value
    # kerchunk imports the symbol by name (``from zarr.meta import ...``), so rebind
    # it in the modules that captured the original.
    for modname in ("kerchunk.hdf", "kerchunk.netCDF3"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if getattr(mod, "encode_fill_value", None) is orig:
            mod.encode_fill_value = encode_fill_value


_patch_zarr_fill_value()

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
    """True if kerchunk can build a reference for this file's format."""
    return detect_format(path) in ("hdf5", "netcdf3")


def reference_one(path: str) -> dict:
    """Build a per-file kerchunk reference using the maker for its format."""
    fmt = detect_format(path)
    if fmt == "hdf5":
        return SingleHdf5ToZarr(path, inline_threshold=0).translate()
    if fmt == "netcdf3":
        # inline_threshold=0 disables array inlining, which otherwise crashes on
        # scalar (0-dim) variables such as a grid_mapping container.
        return NetCDF3ToZarr(path, inline_threshold=0).translate()
    raise ValueError(
        f"Cannot build a kerchunk reference for `{path}` (unrecognized format)."
    )


# Sentinel default: defer scheduler selection to dask, like xarray does.
SCHEDULER_AUTO = "auto"


def _generate_references(
    files: Sequence[str], *, scheduler=SCHEDULER_AUTO
) -> List[dict]:
    """Build the per-file references for ``files``, returned in input order.

    Each :func:`reference_one` reads one file's metadata independently of the
    others, so this is the embarrassingly-parallel hot spot of a cold build.
    ``scheduler`` controls how those reads run:

    * ``"auto"`` (default): mirror xarray and defer to dask's resolution. If a
      scheduler is active (a distributed ``Client`` set as default, or one in
      ``dask.config``) it is picked up automatically; with nothing active we stay
      serial in-process rather than spin up dask's threaded default, which is
      GIL-bound for HDF5 metadata parsing and would only add overhead.
    * ``None``: force serial in-process (no dask involved).
    * anything else (``"processes"``, ``"threads"``, ``"synchronous"``, a
      distributed ``Client``, a get-callable): forwarded verbatim to
      :func:`dask.compute` as its ``scheduler`` — use ``"processes"`` to dodge the
      h5py/HDF5 GIL without a cluster.

    ``reference_one`` is resolved on the module at call time, so a monkeypatch (and
    the in-process schedulers) see the override; with ``"processes"`` the workers
    re-import this module, re-applying the fill-value shim.
    """
    if not files:
        return []
    if scheduler is None:
        return [reference_one(f) for f in files]
    import dask
    from dask.base import get_scheduler

    if scheduler == SCHEDULER_AUTO and get_scheduler() is None:
        return [reference_one(f) for f in files]
    tasks = [dask.delayed(reference_one)(f) for f in files]
    compute_kwargs = {} if scheduler == SCHEDULER_AUTO else {"scheduler": scheduler}
    return list(dask.compute(*tasks, **compute_kwargs))


# ------------------------------------------------------------- combine / open

def open_reference(ref: Mapping) -> xr.Dataset:
    """Open a kerchunk reference as a lazy, CF-decoded xarray Dataset."""
    fs = fsspec.filesystem("reference", fo=ref)
    mapper = fs.get_mapper("")
    return xr.open_dataset(
        mapper, engine="zarr", consolidated=False, decode_coords="all", chunks={}
    )


def _combine(datasets: Sequence[xr.Dataset], combine: str, concat_dim) -> xr.Dataset:
    """Recombine per-file datasets the way the legacy direct open would.

    ``by_coords`` orders by coordinate values (robust to file naming and to mixed
    NetCDF3/NetCDF4 formats); ``nested`` stacks in the given (already coordinate- or
    filename-sorted) order along ``concat_dim`` for bare index axes without a
    coordinate.
    """
    if len(datasets) == 1:
        return datasets[0]
    if combine == COMBINE_NESTED:
        return xr.combine_nested(
            list(datasets), concat_dim=concat_dim, combine_attrs="override"
        )
    return xr.combine_by_coords(list(datasets), combine_attrs="override")


# --------------------------------------------------------------- store payloads

def _assemble_store(
    file_refs: Sequence[Tuple[str, dict]], *, combine: str, concat_dim
) -> dict:
    """Package per-file refs + combine spec into the combined store payload."""
    return {
        "kerchunk_version": KERCHUNK_VERSION,
        "combine": combine,
        "concat_dim": concat_dim,
        "file_refs": [r for _, r in file_refs],
    }


def build_store(
    files: Sequence[str],
    *,
    combine: str = COMBINE_BY_COORDS,
    concat_dim=None,
    scheduler=SCHEDULER_AUTO,
) -> Optional[dict]:
    """In-memory store payload (all per-file refs built fresh).

    ``scheduler`` parallelizes the per-file reference build (see
    :func:`_generate_references`). Returns ``None`` if any file is not
    kerchunk-referenceable (caller falls back to ``open_mfdataset``).
    """
    if not all(is_referenceable(f) for f in files):
        return None
    refs = _generate_references(files, scheduler=scheduler)
    file_refs = list(zip(files, refs))
    return _assemble_store(file_refs, combine=combine, concat_dim=concat_dim)


def _ref_file(cache_dir: str, posix_path: str) -> str:
    h = hashlib.sha1(posix_path.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, FILES_SUBDIR, h + ".json")


def _prune_ref_files(cache_dir: str, keep_posix: set) -> None:
    """Remove per-file reference JSONs no longer in the current file set."""
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


def cached_build_store(
    files: Sequence[str],
    cache_dir: str,
    *,
    reuse_keys: Sequence[str] = (),
    combine: str = COMBINE_BY_COORDS,
    concat_dim=None,
    scheduler=SCHEDULER_AUTO,
) -> Optional[dict]:
    """Incrementally (re)build the on-disk store under ``cache_dir``.

    Per-file references whose POSIX path is in ``reuse_keys`` are loaded from disk;
    the rest (the *stale* set) are regenerated — that build is the parallelizable
    step and honours ``scheduler`` (see :func:`_generate_references`). Newly built
    references are persisted, stale per-file refs are pruned and the combined
    ``store.json`` is written. Returns the payload, or ``None`` (with no writes) if a
    file is not referenceable — the caller then falls back to ``open_mfdataset``.
    """
    if not all(is_referenceable(f) for f in files):
        return None
    reuse = set(reuse_keys)
    posix_paths = [_make_path_posix(f) for f in files]
    # Generate the stale references up front (optionally in parallel); reused ones
    # are read back from their per-file JSON below. Writes stay on the driver, in
    # input order, so each ``files/<hash>.json`` has a single, atomic writer.
    stale_idx = [i for i, p in enumerate(posix_paths) if p not in reuse]
    fresh = dict(
        zip(
            stale_idx,
            _generate_references([files[i] for i in stale_idx], scheduler=scheduler),
        )
    )
    file_refs = []
    for i, f in enumerate(files):
        rp = _ref_file(cache_dir, posix_paths[i])
        if i in fresh:
            ref = fresh[i]
            _cache.write_json(rp, ref)
        else:  # reused: load from disk, regenerating on a (rare) cache miss
            ref = _cache.read_json(rp)
            if ref is None:
                ref = reference_one(f)
                _cache.write_json(rp, ref)
        file_refs.append((f, ref))
    payload = _assemble_store(file_refs, combine=combine, concat_dim=concat_dim)
    _prune_ref_files(cache_dir, set(posix_paths))
    _cache.write_json(os.path.join(cache_dir, STORE_FILE), payload)
    return payload


def load_store(cache_dir: str) -> Optional[dict]:
    """Load the combined store payload from ``cache_dir`` (or ``None``)."""
    return _cache.read_json(os.path.join(cache_dir, STORE_FILE))


def open_store(payload: Mapping) -> xr.Dataset:
    """Reopen the combined lazy dataset from a :func:`build_store` payload."""
    datasets = [open_reference(r) for r in payload["file_refs"]]
    return _combine(
        datasets,
        payload.get("combine", COMBINE_BY_COORDS),
        payload.get("concat_dim"),
    )
