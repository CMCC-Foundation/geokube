"""Pure-python unit tests for the metadata-cache backbone (geokube.backend._cache).

These exercise the manifest / change-detection / JSON-index / atomic-IO logic and
do NOT need the heavy I/O stack (kerchunk / ESMF). They use tiny dummy files.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from geokube.backend import _cache


def _touch(path, content=b"x"):
    with open(path, "wb") as f:
        f.write(content)
    return str(path)


def _bump_mtime(path, seconds=10):
    st = os.stat(path)
    new = st.st_mtime_ns + seconds * 1_000_000_000
    os.utime(path, ns=(new, new))


# ------------------------------------------------------------------- manifest

def test_build_manifest_structure(tmp_path):
    a = _touch(tmp_path / "a.nc", b"aaaa")
    b = _touch(tmp_path / "b.nc", b"bb")
    m = _cache.build_manifest([a, b], context={"pattern": "{var}.nc"})
    assert m["cache_format_version"] == _cache.CACHE_FORMAT_VERSION
    assert m["geokube_version"] and m["xarray_version"]
    assert m["context"] == {"pattern": "{var}.nc"}
    # posix-normalized keys, [mtime_ns, size] signatures
    assert set(m["files"]) == {a.replace("\\", "/"), b.replace("\\", "/")}
    for sig in m["files"].values():
        assert isinstance(sig, list) and len(sig) == 2
    assert m["files"][a.replace("\\", "/")][1] == 4  # size of b"aaaa"


def test_manifest_matches_identical(tmp_path):
    a = _touch(tmp_path / "a.nc")
    ctx = {"pattern": "p"}
    m1 = _cache.build_manifest([a], context=ctx)
    m2 = _cache.build_manifest([a], context=ctx)
    assert _cache.manifest_matches(m1, m2)


def test_manifest_detects_changed_file(tmp_path):
    a = _touch(tmp_path / "a.nc")
    cached = _cache.build_manifest([a], context={})
    _bump_mtime(a)
    current = _cache.build_manifest([a], context={})
    assert not _cache.manifest_matches(cached, current)
    d = _cache.manifest_diff(cached, current)
    assert d["changed"] == [a.replace("\\", "/")]
    assert d["added"] == [] and d["removed"] == []


def test_manifest_detects_added_and_removed(tmp_path):
    a = _touch(tmp_path / "a.nc")
    b = _touch(tmp_path / "b.nc")
    cached = _cache.build_manifest([a], context={})
    current = _cache.build_manifest([a, b], context={})
    assert _cache.manifest_diff(cached, current)["added"] == [b.replace("\\", "/")]
    assert not _cache.manifest_matches(cached, current)
    # removal
    current2 = _cache.build_manifest([], context={})
    assert _cache.manifest_diff(cached, current2)["removed"] == [a.replace("\\", "/")]


def test_manifest_context_change_invalidates(tmp_path):
    a = _touch(tmp_path / "a.nc")
    m1 = _cache.build_manifest([a], context={"pattern": "p1"})
    m2 = _cache.build_manifest([a], context={"pattern": "p2"})
    assert not _cache.context_matches(m1, m2)
    assert not _cache.manifest_matches(m1, m2)


def test_manifest_version_mismatch_invalidates(tmp_path):
    a = _touch(tmp_path / "a.nc")
    current = _cache.build_manifest([a], context={})
    stale = dict(current)
    stale["geokube_version"] = "0.0.0-old"
    assert not _cache.context_matches(stale, current)


def test_manifest_survives_json_roundtrip(tmp_path):
    a = _touch(tmp_path / "a.nc")
    ctx = {"pattern": "p", "concat_dims": ["time"], "mapping": {"x": {"name": "y"}}}
    cached = _cache.build_manifest([a], context=ctx)
    p = tmp_path / "manifest.json"
    _cache.write_json(str(p), cached)
    loaded = _cache.read_json(str(p))
    current = _cache.build_manifest([a], context=ctx)
    assert _cache.manifest_matches(loaded, current)


# ------------------------------------------------------------------- JSON I/O

def test_write_read_json_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    _cache.write_json(str(p), payload)
    assert json.loads(p.read_text()) == payload
    assert _cache.read_json(str(p)) == payload
    # no leftover temp files
    assert [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")] == []


def test_write_json_encodes_numpy_and_datetime(tmp_path):
    p = tmp_path / "x.json"
    payload = {"i": np.int64(7), "f": np.float32(1.5), "t": pd.Timestamp("2020-01-02")}
    _cache.write_json(str(p), payload)
    loaded = _cache.read_json(str(p))
    assert loaded["i"] == 7 and loaded["f"] == 1.5
    assert loaded["t"].startswith("2020-01-02")


def test_read_json_misses(tmp_path):
    assert _cache.read_json(str(tmp_path / "nope.json")) is None
    # legacy binary pickle -> miss, not a crash
    pk = tmp_path / "legacy.pkl"
    with open(pk, "wb") as f:
        pickle.dump({"some": "object"}, f)
    assert _cache.read_json(str(pk)) is None
    # valid JSON but not a dict -> miss
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2, 3]")
    assert _cache.read_json(str(lst)) is None


# -------------------------------------------------------------- cache directory

def test_ensure_cache_dir_creates(tmp_path):
    d = tmp_path / "cache"
    assert _cache.ensure_cache_dir(str(d)) == str(d)
    assert os.path.isdir(d)
    # idempotent on an existing dir
    assert _cache.ensure_cache_dir(str(d)) == str(d)


def test_ensure_cache_dir_rejects_legacy_file(tmp_path):
    legacy = tmp_path / "old_cache"
    with open(legacy, "wb") as f:
        pickle.dump({"x": 1}, f)
    with pytest.raises(_cache.LegacyCacheFileError):
        _cache.ensure_cache_dir(str(legacy))


# ------------------------------------------------------------------ file-index

def test_index_records_roundtrip():
    df = pd.DataFrame(
        {"var": ["t2m", "tp"], _cache_files_col(): [["a.nc"], ["b.nc", "c.nc"]]}
    ).set_index("var")
    records = _cache.index_to_records(df, ["var"], _cache_files_col())
    assert records == [
        {"var": "t2m", _cache_files_col(): ["a.nc"]},
        {"var": "tp", _cache_files_col(): ["b.nc", "c.nc"]},
    ]
    rebuilt = _cache.records_to_indexed_df(records, ["var"], _cache_files_col())
    assert list(rebuilt.index) == ["t2m", "tp"]
    assert rebuilt.loc["tp", _cache_files_col()] == ["b.nc", "c.nc"]


def _cache_files_col():
    return "files"
