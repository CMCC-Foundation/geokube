"""Kerchunk reference payloads for lazy multi-file ``DataCube`` caching.

Opening hundreds/thousands of NetCDF files with ``xr.open_mfdataset`` can take
hours: the cost is the *open* (per-file metadata read + coordinate alignment), not
the data processing. This module persists that combined-open result as **kerchunk
reference JSON** instead of a pickle, so subsequent opens are cheap and the cache is
portable, secure and robust across library versions.

Design (validated by the Phase-0 spike):

* Each file gets a per-file reference via the maker matching its on-disk format
  (HDF5/NetCDF4 -> :class:`SingleHdf5ToZarr`; NetCDF3-classic ->
  :class:`NetCDF3ToZarr`), chosen by magic-byte sniffing.
* Files are **partitioned by encoding signature** (dtype, chunks, compressor,
  filters per variable). A single Zarr ``.zarray`` cannot describe heterogeneous
  layouts, so e.g. uncompressed/contiguous NetCDF3 and chunked/compressed NetCDF4
  fall into different partitions.
* Each homogeneous partition is concatenated with :class:`MultiZarrToZarr` into one
  combined reference; the (few) partitions are recombined at open time with
  :func:`xarray.combine_by_coords` — which infers everything at the decoded-array
  level, so mixed-format groups Just Work without pickle.

The on-disk payload (one JSON per partition, see ``_cache`` layout) is produced by
:func:`build_store` and reopened by :func:`open_store`.
"""
from __future__ import annotations

__all__ = [
    "KERCHUNK_VERSION",
    "detect_format",
    "is_referenceable",
    "reference_one",
    "zarray_signature",
    "signature_key",
    "partition",
    "infer_concat_dims",
    "infer_identical_dims",
    "combine_partition",
    "open_reference",
    "build_store",
    "cached_build_store",
    "load_store",
    "open_store",
]

import hashlib
import importlib
import json
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import fsspec
import xarray as xr

import kerchunk
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.netCDF3 import NetCDF3ToZarr
from kerchunk.combine import MultiZarrToZarr

from geokube.backend import _cache
from geokube.utils.format_parsing import _make_path_posix

KERCHUNK_VERSION = kerchunk.__version__

# Cache-directory layout for the per-cube kerchunk store.
FILES_SUBDIR = "files"      # one per-file reference JSON (enables incremental rebuild)
STORE_FILE = "store.json"   # the combined per-partition references


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


# ------------------------------------------------------------------ signatures

def _refs_dict(ref: Mapping) -> Mapping:
    return ref.get("refs", ref)


def _var_names(ref: Mapping) -> List[str]:
    return sorted(
        k[: -len("/.zarray")] for k in _refs_dict(ref) if k.endswith("/.zarray")
    )


def _array_dims(ref: Mapping, var: str) -> List[str]:
    zattrs = _refs_dict(ref).get(f"{var}/.zattrs")
    if zattrs is None:
        return []
    z = json.loads(zattrs) if isinstance(zattrs, str) else zattrs
    return list(z.get("_ARRAY_DIMENSIONS", []))


def zarray_signature(ref: Mapping) -> dict:
    """Per-variable encoding signature: ``{var: [dtype, chunks, compressor, filters]}``."""
    sig = {}
    for k, v in _refs_dict(ref).items():
        if k.endswith("/.zarray"):
            var = k[: -len("/.zarray")]
            za = json.loads(v) if isinstance(v, str) else v
            sig[var] = [
                za.get("dtype"),
                za.get("chunks"),
                za.get("compressor"),
                za.get("filters"),
            ]
    return sig


def signature_key(ref: Mapping) -> str:
    """Stable hashable key for a reference's encoding signature."""
    return json.dumps(zarray_signature(ref), sort_keys=True)


def partition(
    files_refs: Sequence[Tuple[str, dict]]
) -> List[Tuple[str, List[Tuple[str, dict]]]]:
    """Group ``(path, ref)`` pairs by identical encoding signature (first-seen order)."""
    groups: dict = {}
    order: List[str] = []
    for path, ref in files_refs:
        key = signature_key(ref)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((path, ref))
    return [(k, groups[k]) for k in order]


# ----------------------------------------------------------------- dim inference

def infer_identical_dims(ref: Mapping, concat_dims: Sequence[str]) -> List[str]:
    """Variables that do NOT span any concat dim -> identical across files."""
    cd = set(concat_dims)
    return [v for v in _var_names(ref) if not (set(_array_dims(ref, v)) & cd)]


def infer_concat_dims(refs: Sequence[dict]) -> Optional[List[str]]:
    """Best-effort concat dims: dimension coords that DIFFER across files.

    Opens the first two references (cheap, metadata only). Returns ``[]`` for a
    single file, the differing dimension-coord names, or ``None`` when nothing
    varies (ambiguous -> caller should fall back to ``open_mfdataset``).
    """
    if len(refs) < 2:
        return []
    ds0 = open_reference(refs[0])
    ds1 = open_reference(refs[1])
    diff = []
    for d in ds0.sizes:
        if d in ds0.coords and d in ds1.coords:
            v0 = np.asarray(ds0[d].values)
            v1 = np.asarray(ds1[d].values)
            if v0.shape != v1.shape or not np.array_equal(v0, v1):
                diff.append(d)
    return diff or None


# ------------------------------------------------------------- combine / open

def combine_partition(
    refs: Sequence[dict],
    concat_dims: Sequence[str],
    identical_dims: Optional[Sequence[str]] = None,
) -> dict:
    """MultiZarrToZarr a homogeneous partition (or return the lone ref)."""
    if len(refs) == 1:
        return refs[0]
    if identical_dims is None:
        identical_dims = infer_identical_dims(refs[0], concat_dims)
    return MultiZarrToZarr(
        list(refs),
        concat_dims=list(concat_dims),
        identical_dims=list(identical_dims),
    ).translate()


def open_reference(ref: Mapping) -> xr.Dataset:
    """Open a kerchunk reference as a lazy, CF-decoded xarray Dataset."""
    fs = fsspec.filesystem("reference", fo=ref)
    mapper = fs.get_mapper("")
    return xr.open_dataset(
        mapper, engine="zarr", consolidated=False, decode_coords="all", chunks={}
    )


# --------------------------------------------------------------- store payloads

def _assemble_store(
    file_refs: Sequence[Tuple[str, dict]],
    concat_dims: Optional[Sequence[str]],
    identical_dims: Optional[Sequence[str]],
) -> Optional[dict]:
    """Partition + concatenate per-file refs into the combined store payload.

    Returns ``None`` if ``concat_dims`` is not given and cannot be inferred.
    """
    if concat_dims is None:
        concat_dims = infer_concat_dims([r for _, r in file_refs])
        if concat_dims is None:
            return None
    partitions = []
    for sig_key, items in partition(file_refs):
        combined = combine_partition(
            [r for _, r in items], concat_dims, identical_dims
        )
        partitions.append(
            {
                "signature": sig_key,
                "files": [p for p, _ in items],
                "ref": combined,
            }
        )
    return {
        "kerchunk_version": KERCHUNK_VERSION,
        "concat_dims": list(concat_dims),
        "identical_dims": list(identical_dims) if identical_dims else None,
        "partitions": partitions,
    }


def build_store(
    files: Sequence[str],
    *,
    concat_dims: Optional[Sequence[str]] = None,
    identical_dims: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """In-memory store payload (all per-file refs built fresh).

    Returns ``None`` if any file is not kerchunk-referenceable, or if concat dims
    cannot be determined (caller falls back to ``open_mfdataset``).
    """
    if not all(is_referenceable(f) for f in files):
        return None
    file_refs = [(f, reference_one(f)) for f in files]
    return _assemble_store(file_refs, concat_dims, identical_dims)


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
    concat_dims: Optional[Sequence[str]] = None,
    identical_dims: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Incrementally (re)build the on-disk store under ``cache_dir``.

    Per-file references whose POSIX path is in ``reuse_keys`` are loaded from disk;
    the rest are regenerated and persisted. Stale per-file refs are pruned and the
    combined ``store.json`` is written. Returns the payload, or ``None`` (with no
    writes) if a file is not referenceable / concat dims are indeterminate — the
    caller then falls back to ``open_mfdataset``.
    """
    if not all(is_referenceable(f) for f in files):
        return None
    reuse = set(reuse_keys)
    file_refs = []
    for f in files:
        posix = _make_path_posix(f)
        rp = _ref_file(cache_dir, posix)
        ref = _cache.read_json(rp) if posix in reuse else None
        if ref is None:
            ref = reference_one(f)
            _cache.write_json(rp, ref)
        file_refs.append((f, ref))
    payload = _assemble_store(file_refs, concat_dims, identical_dims)
    if payload is None:
        return None
    _prune_ref_files(cache_dir, {_make_path_posix(f) for f in files})
    _cache.write_json(os.path.join(cache_dir, STORE_FILE), payload)
    return payload


def load_store(cache_dir: str) -> Optional[dict]:
    """Load the combined store payload from ``cache_dir`` (or ``None``)."""
    return _cache.read_json(os.path.join(cache_dir, STORE_FILE))


def open_store(payload: Mapping) -> xr.Dataset:
    """Reopen the combined lazy dataset from a :func:`build_store` payload."""
    datasets = [open_reference(p["ref"]) for p in payload["partitions"]]
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs="override")
