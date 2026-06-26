"""On-disk metadata cache: manifest (change-detection), JSON file-index, atomic I/O.

This module is the format-agnostic backbone of geokube's metadata caching. It has
no dependency on the heavy I/O stack (kerchunk / ESMF / xarray backends) so it can
be unit-tested in plain Python. The kerchunk reference payloads live in
``_kerchunk.py``; here we only deal with:

* a **manifest** mapping every input file to a ``(st_mtime_ns, st_size)`` signature
  plus the invalidation context (geokube/xarray versions + caller context such as
  ``pattern`` / ``id_pattern`` / ``mapping`` / ``concat_dims``). Comparing the cached
  manifest with a freshly built one tells us whether the cache is still valid, and
  the per-file diff drives *incremental* regeneration;
* a JSON **file-index** (dataset-attribute combos -> list of files) for
  ``open_dataset``;
* tolerant, **atomic** JSON read/write helpers and the cache-directory layout.

``metadata_cache_path`` is a *directory* (see ``ensure_cache_dir``); a legacy pickle
file found at that path raises :class:`LegacyCacheFileError`.
"""
from __future__ import annotations

__all__ = [
    "CACHE_FORMAT_VERSION",
    "MANIFEST_FILE",
    "INDEX_FILE",
    "CUBES_DIR",
    "LegacyCacheFileError",
    "ensure_cache_dir",
    "build_manifest",
    "context_matches",
    "manifest_diff",
    "manifest_matches",
    "index_to_records",
    "records_to_indexed_df",
    "read_json",
    "write_json",
]

import json
import os
import tempfile
from datetime import date, datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from geokube.version import __version__ as GEOKUBE_VERSION
from geokube.utils.format_parsing import _make_path_posix

# Bump when the on-disk schema changes; old caches then read as a miss. v2 =
# per-partition consolidated kerchunk store (was a flat per-file ``file_refs`` list).
CACHE_FORMAT_VERSION = 2

# Cache-directory layout (relative to ``metadata_cache_path``).
MANIFEST_FILE = "manifest.json"
INDEX_FILE = "index.json"
CUBES_DIR = "cubes"


class LegacyCacheFileError(ValueError):
    """Raised when ``metadata_cache_path`` points to a legacy pickle *file*.

    The cache path is now a *directory*; a regular file at that location is most
    likely a pre-2026.06 pickle cache and must be removed (or a directory path
    passed instead) before the new cache can be used.
    """


# --------------------------------------------------------------------------- IO

def _json_default(obj: Any) -> Any:
    """Serialize numpy scalars / arrays and datetimes that ``json`` rejects."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Atomically write ``payload`` as JSON to ``path`` (temp file + ``os.replace``)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str) -> Optional[dict]:
    """Return the parsed JSON dict, or ``None`` on any problem.

    Missing file, decode error (e.g. a legacy *binary* pickle), or a non-dict
    payload are all treated as a cache miss rather than raising.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def ensure_cache_dir(path: str) -> str:
    """Validate/create the cache *directory* at ``path``; return it.

    Raises :class:`LegacyCacheFileError` if ``path`` is an existing regular file.
    """
    if os.path.isfile(path):
        raise LegacyCacheFileError(
            f"metadata_cache_path is now a directory, but a regular file was found"
            f" at `{path}`. It is most likely a legacy pickle cache: remove it or"
            f" pass a directory path."
        )
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------- manifest

def _stat_signature(path: str) -> list:
    """``[st_mtime_ns, st_size]`` for ``path`` (raises ``OSError`` if missing)."""
    st = os.stat(path)
    return [st.st_mtime_ns, st.st_size]


def _normalize(value: Any) -> Any:
    """Round-trip ``value`` through JSON so dicts/scalars compare stably."""
    return json.loads(json.dumps(value, default=_json_default))


def build_manifest(
    files: Sequence[str],
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the cache manifest for ``files``.

    ``context`` carries caller-specific invalidation keys (e.g. ``pattern``,
    ``id_pattern``, ``mapping``, ``concat_dims``, ``kerchunk_version``). Files are
    stat'd here; paths are POSIX-normalized for stable, OS-independent keys.
    """
    file_sigs = {}
    for f in files:
        file_sigs[_make_path_posix(f)] = _stat_signature(f)
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "geokube_version": GEOKUBE_VERSION,
        "xarray_version": xr.__version__,
        "context": _normalize(dict(context or {})),
        "files": file_sigs,
    }


def context_matches(cached: Optional[Mapping], current: Mapping) -> bool:
    """True iff versions/format/context agree (ignores the per-file set)."""
    if not isinstance(cached, Mapping):
        return False
    keys = ("cache_format_version", "geokube_version", "xarray_version", "context")
    return all(cached.get(k) == current.get(k) for k in keys)


def manifest_diff(cached: Optional[Mapping], current: Mapping) -> dict:
    """Per-file diff ``{added, removed, changed}`` (POSIX path keys).

    ``added``: in ``current`` not in ``cached``. ``removed``: vice versa.
    ``changed``: present in both with a different ``[mtime_ns, size]`` signature.
    """
    cached_files = dict((cached or {}).get("files", {}))
    current_files = dict(current.get("files", {}))
    ck, curk = set(cached_files), set(current_files)
    added = sorted(curk - ck)
    removed = sorted(ck - curk)
    changed = sorted(k for k in (ck & curk) if cached_files[k] != current_files[k])
    return {"added": added, "removed": removed, "changed": changed}


def manifest_matches(cached: Optional[Mapping], current: Mapping) -> bool:
    """True iff the cache is fully valid: same context AND identical file set."""
    if not context_matches(cached, current):
        return False
    d = manifest_diff(cached, current)
    return not (d["added"] or d["removed"] or d["changed"])


# ------------------------------------------------------------------ file-index

def index_to_records(
    df_indexed: pd.DataFrame,
    attrs: Sequence[str],
    files_col: str,
) -> list:
    """Serialize the indexed file-index DataFrame to JSON-safe row dicts.

    ``df_indexed`` is indexed by ``attrs`` with a ``files_col`` column holding a
    list of file paths. Returns ``[{<attr>: val, ..., files_col: [paths]}]``.
    """
    flat = df_indexed.reset_index()
    cols = list(attrs) + [files_col]
    return flat[cols].to_dict("records")


def records_to_indexed_df(
    records: Sequence[Mapping],
    attrs: Sequence[str],
    files_col: str,
) -> pd.DataFrame:
    """Inverse of :func:`index_to_records`: rebuild the ``attrs``-indexed DataFrame."""
    df = pd.DataFrame(list(records))
    if list(attrs):
        df = df.set_index(list(attrs))
    return df
