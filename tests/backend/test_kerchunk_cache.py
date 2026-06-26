"""Integration tests for the kerchunk metadata cache with the reader/writer split.

The catalog (writer) builds the cache via ``build_metadata_cache``; the API (reader)
consumes it via ``open_datacube`` / ``open_dataset`` with ``metadata_caching=True``,
which never write and raise ``CacheNotExist`` if the cache is absent. Marked
``integration`` (needs the kerchunk/netCDF stack of the base image; real I/O).
"""
import os

import numpy as np
import pytest
import xarray as xr

from geokube import open_datacube, open_dataset, build_metadata_cache
from geokube.backend import _cache, _kerchunk
from geokube.core.errs import CacheNotExist

SRC = os.path.join("tests", "resources", "rlat-rlon-tmin2m.nc")
CDIM = "time"


def _make_slabs(tmp_path, n=4, formats=("nc4",)):
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()  # eager: avoids HDF5/dask write deadlock under emulation
    length = src.sizes[CDIM]
    edges = [round(i * length / n) for i in range(n + 1)]
    out = {f: [] for f in formats}
    for i in range(n):
        sl = src.isel({CDIM: slice(edges[i], edges[i + 1])})
        if "nc4" in formats:
            p = tmp_path / f"slab{i}.nc"
            sl.to_netcdf(p, engine="netcdf4", format="NETCDF4")
            out["nc4"].append(str(p))
        if "nc3" in formats:
            sl3 = sl.copy()
            for v in list(sl3.variables):
                if sl3[v].dtype == np.int64:
                    sl3[v] = sl3[v].astype(np.int32)
                for k in ("zlib", "complevel", "chunksizes"):
                    sl3[v].encoding.pop(k, None)
            p3 = tmp_path / f"slab{i}_n3.nc"
            sl3.to_netcdf(p3, format="NETCDF3_CLASSIC")
            out["nc3"].append(str(p3))
    return out


def _assert_cube_values_match(c1, c2):
    a, b = c1.to_xarray(), c2.to_xarray()
    assert set(a.data_vars) == set(b.data_vars)
    for v in a.data_vars:
        xr.testing.assert_allclose(a[v].sortby(CDIM), b[v].sortby(CDIM))


# ------------------------------------------------------------- datacube: build/read

@pytest.mark.integration
def test_datacube_build_layout_and_read_equivalence(tmp_path):
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    summary = build_metadata_cache(
        files, metadata_cache_path=str(cache), concat_dims=[CDIM]
    )
    assert summary["built"] == 1 and summary["skipped"] == []
    assert os.path.isfile(cache / _cache.MANIFEST_FILE)
    assert os.path.isfile(cache / _kerchunk.STORE_FILE)
    assert len(os.listdir(cache / _kerchunk.FILES_SUBDIR)) == len(files)

    cube = open_datacube(files, metadata_caching=True,
                         metadata_cache_path=str(cache), concat_dims=[CDIM])
    _assert_cube_values_match(cube, open_datacube(files, concat_dims=[CDIM]))


@pytest.mark.integration
def test_datacube_reader_raises_if_absent(tmp_path):
    files = _make_slabs(tmp_path)["nc4"]
    with pytest.raises(CacheNotExist):
        open_datacube(files, metadata_caching=True,
                      metadata_cache_path=str(tmp_path / "nope"), concat_dims=[CDIM])


@pytest.mark.integration
def test_reader_performs_no_writes(tmp_path, monkeypatch):
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache), concat_dims=[CDIM])

    def _boom(*a, **k):
        raise AssertionError("reader attempted a write")

    monkeypatch.setattr(_cache, "write_json", _boom)
    monkeypatch.setattr(_cache, "ensure_cache_dir", _boom)
    cube = open_datacube(files, metadata_caching=True,
                         metadata_cache_path=str(cache), concat_dims=[CDIM])
    assert cube is not None


@pytest.mark.integration
def test_build_is_incremental(tmp_path, monkeypatch):
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache), concat_dims=[CDIM])

    # Rebuild with nothing changed -> no per-file reference is regenerated.
    def _boom(*a, **k):
        raise AssertionError("reference_one called on an unchanged rebuild")

    monkeypatch.setattr(_kerchunk, "reference_one", _boom)
    build_metadata_cache(files, metadata_cache_path=str(cache), concat_dims=[CDIM])
    monkeypatch.undo()

    # Touch one file -> only that file's reference is regenerated.
    st = os.stat(files[1])
    os.utime(files[1], ns=(st.st_mtime_ns + 10**9, st.st_mtime_ns + 10**9))
    calls = []
    real = _kerchunk.reference_one
    monkeypatch.setattr(
        _kerchunk, "reference_one",
        lambda p, *a, **k: (calls.append(p), real(p, *a, **k))[1],
    )
    build_metadata_cache(files, metadata_cache_path=str(cache), concat_dims=[CDIM])
    assert calls == [files[1]]


@pytest.mark.integration
def test_build_explicit_scheduler_matches_serial(tmp_path):
    # An explicit dask scheduler must produce byte-identical references (and a
    # matching cube) to the serial build. "synchronous" exercises the delayed/
    # compute plumbing deterministically, without real-concurrency h5py races.
    files = _make_slabs(tmp_path)["nc4"]
    serial, par = tmp_path / "serial", tmp_path / "par"
    build_metadata_cache(files, metadata_cache_path=str(serial), scheduler=None)
    build_metadata_cache(files, metadata_cache_path=str(par), scheduler="synchronous")

    s_store, p_store = _kerchunk.load_store(str(serial)), _kerchunk.load_store(str(par))
    assert p_store["file_refs"] == s_store["file_refs"]  # same refs, in input order
    assert p_store["combine"] == s_store["combine"]
    cube = open_datacube(files, metadata_caching=True, metadata_cache_path=str(par))
    _assert_cube_values_match(cube, open_datacube(files))


@pytest.mark.integration
def test_build_auto_attaches_to_active_scheduler(tmp_path, monkeypatch):
    # scheduler="auto" (default) defers to dask like xarray: with a scheduler active
    # it routes the build through dask.compute; with none active it stays serial.
    import dask

    files = _make_slabs(tmp_path)["nc4"]
    serial = tmp_path / "serial"
    build_metadata_cache(files, metadata_cache_path=str(serial), scheduler=None)

    real_compute, calls = dask.compute, []
    monkeypatch.setattr(
        dask, "compute", lambda *a, **k: (calls.append(k), real_compute(*a, **k))[1]
    )
    with dask.config.set(scheduler="synchronous"):
        build_metadata_cache(files, metadata_cache_path=str(tmp_path / "auto"))
    assert calls  # an active scheduler was picked up automatically
    assert (
        _kerchunk.load_store(str(tmp_path / "auto"))["file_refs"]
        == _kerchunk.load_store(str(serial))["file_refs"]
    )


@pytest.mark.integration
def test_datacube_mixed_format_combine(tmp_path):
    # by_coords recombines at the decoded-array level, so mixed NetCDF3/NetCDF4
    # files combine into one cube without any partitioning.
    slabs = _make_slabs(tmp_path, formats=("nc4", "nc3"))
    mixed = slabs["nc4"][:2] + slabs["nc3"][2:]
    cache = tmp_path / "cache"
    build_metadata_cache(mixed, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["file_refs"]) == len(mixed)
    assert store["combine"] == "by_coords"
    cube = open_datacube(mixed, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(mixed))


@pytest.mark.integration
def test_datacube_default_combine_by_coords(tmp_path):
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    assert _kerchunk.load_store(str(cache))["combine"] == "by_coords"
    cube = open_datacube(files, metadata_caching=True,
                         metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(files))


@pytest.mark.integration
def test_datacube_nested_combine(tmp_path):
    # `nested` stacks in file order along an explicit concat_dim (for bare index
    # axes without a coordinate); the spec is persisted and replayed by the reader.
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache),
                         combine="nested", concat_dim=CDIM)
    store = _kerchunk.load_store(str(cache))
    assert store["combine"] == "nested" and store["concat_dim"] == CDIM
    cube = open_datacube(files, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(files))


@pytest.mark.integration
def test_build_skips_unreferenceable(tmp_path, monkeypatch):
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    monkeypatch.setattr(_kerchunk, "is_referenceable", lambda p: False)
    summary = build_metadata_cache(files, metadata_cache_path=str(cache),
                                   concat_dims=[CDIM])
    assert summary["built"] == 0 and summary["skipped"]
    assert not os.path.exists(cache / _kerchunk.STORE_FILE)
    monkeypatch.undo()
    # Nothing was published -> the reader errors.
    with pytest.raises(CacheNotExist):
        open_datacube(files, metadata_caching=True,
                      metadata_cache_path=str(cache), concat_dims=[CDIM])


@pytest.mark.integration
def test_legacy_file_path_raises(tmp_path):
    files = _make_slabs(tmp_path)["nc4"]
    legacy = tmp_path / "legacy_cache"
    legacy.write_bytes(b"\x80\x04legacy-pickle")
    with pytest.raises(_cache.LegacyCacheFileError):
        build_metadata_cache(files, metadata_cache_path=str(legacy),
                             concat_dims=[CDIM])
    with pytest.raises(_cache.LegacyCacheFileError):
        open_datacube(files, metadata_caching=True,
                      metadata_cache_path=str(legacy), concat_dims=[CDIM])


# ---------------------------------------------------------- datacube: single file
# The cache applies uniformly to a standalone single-file cube (bare string path),
# not just multi-file lists/globs. The open-time payoff for one file is negligible;
# the value is a uniform code path and having the mechanism ready if the dataset
# later grows to many files. Same read-only contract: CacheNotExist if absent.

@pytest.mark.integration
def test_single_file_datacube_build_and_read_equivalence(tmp_path):
    cache = tmp_path / "cache"
    summary = build_metadata_cache(SRC, metadata_cache_path=str(cache))
    assert summary["built"] == 1 and summary["skipped"] == []
    assert os.path.isfile(cache / _kerchunk.STORE_FILE)
    assert len(_kerchunk.load_store(str(cache))["file_refs"]) == 1

    cube = open_datacube(SRC, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(SRC))


@pytest.mark.integration
def test_single_file_reader_raises_if_absent(tmp_path):
    with pytest.raises(CacheNotExist):
        open_datacube(SRC, metadata_caching=True,
                      metadata_cache_path=str(tmp_path / "nope"))


@pytest.mark.integration
def test_single_file_list_input_still_builds(tmp_path):
    # A 1-element list must keep working (no regression from the str->[str] fix).
    cache = tmp_path / "cache"
    summary = build_metadata_cache([SRC], metadata_cache_path=str(cache))
    assert summary["built"] == 1
    cube = open_datacube([SRC], metadata_caching=True,
                         metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube([SRC]))


# -------------------------------------------------------------- dataset: build/read

@pytest.mark.integration
def test_open_dataset_build_read_and_layout(tmp_path, monkeypatch):
    glob_path = os.path.join("tests", "resources",
                             "era5-single-levels-reanalysis_*.nc")
    pattern = os.path.join("tests", "resources",
                           "era5-single-levels-reanalysis_{var}.nc")
    cache = tmp_path / "ds_cache"
    summary = build_metadata_cache(glob_path, pattern, metadata_cache_path=str(cache))
    assert summary["groups"] >= 1 and summary["built"] == summary["groups"]
    assert os.path.isfile(cache / _cache.INDEX_FILE)

    # index references existing cube stores, each retaining its data variable(s)
    # (catches the packed-int16 _FillValue drop -> the shim must be in effect).
    index = _cache.read_json(str(cache / _cache.INDEX_FILE))
    assert index["records"]
    cubes_dir = cache / _cache.CUBES_DIR
    for g in os.listdir(cubes_dir):
        store = _kerchunk.load_store(str(cubes_dir / g))
        assert store is not None
        assert len(_kerchunk.open_store(store).data_vars) >= 1

    dset = open_dataset(glob_path, pattern, metadata_caching=True,
                        metadata_cache_path=str(cache))
    assert dset is not None

    # Reader must not rebuild the file-index from the filenames.
    import geokube.backend.netcdf as ncmod
    monkeypatch.setattr(
        ncmod, "_get_df_from_files_list",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("index rebuilt")),
    )
    assert open_dataset(glob_path, pattern, metadata_caching=True,
                        metadata_cache_path=str(cache)) is not None


@pytest.mark.integration
def test_open_dataset_reader_raises_if_absent(tmp_path):
    glob_path = os.path.join("tests", "resources",
                             "era5-single-levels-reanalysis_*.nc")
    pattern = os.path.join("tests", "resources",
                           "era5-single-levels-reanalysis_{var}.nc")
    with pytest.raises(CacheNotExist):
        open_dataset(glob_path, pattern, metadata_caching=True,
                     metadata_cache_path=str(tmp_path / "empty"))


@pytest.mark.integration
def test_open_dataset_requires_cache_path():
    glob_path = os.path.join("tests", "resources",
                             "era5-single-levels-reanalysis_*.nc")
    pattern = os.path.join("tests", "resources",
                           "era5-single-levels-reanalysis_{var}.nc")
    with pytest.raises(ValueError):
        open_dataset(glob_path, pattern, metadata_caching=True,
                     metadata_cache_path=None)


# ------------------------------------------------------ datacube: preprocess hook
# `preprocess` (xr.Dataset -> xr.Dataset) must apply UNIFORMLY: single-file & multi,
# cached & non-cached. It runs on the *combined* raw dataset right before the cube is
# built (not per-file). We use a CF-preserving rename of the data variable so the cube
# still opens, and the rename is observable in `.to_xarray().data_vars`.

DVAR = "TMIN_2M"          # the lone data var in rlat-rlon-tmin2m.nc
DVAR_PP = "TMIN_2M_PP"    # preprocess renames it to this


def _rename_dvar(ds):
    # CF-preserving: only the data variable is renamed; coords / grid_mapping /
    # standard_name on coords are untouched, so DataCube.from_xarray still succeeds.
    return ds.rename({DVAR: DVAR_PP})


@pytest.mark.integration
def test_single_file_noncached_applies_preprocess():
    # Pre-fix this raised TypeError (preprocess forwarded to xr.open_dataset on the
    # single-file path). Post-fix it is popped and applied to the combined dataset.
    cube = open_datacube(SRC, preprocess=_rename_dvar)
    dv = set(cube.to_xarray().data_vars)
    assert DVAR_PP in dv and DVAR not in dv


@pytest.mark.integration
def test_single_file_cached_preprocess_transparency(tmp_path):
    # The exact production WRF scenario: single file, metadata_caching=True. The cache
    # must be transparent -> cached+preprocess == non-cached+preprocess.
    cache = tmp_path / "cache"
    build_metadata_cache(SRC, metadata_cache_path=str(cache))
    cached_pp = open_datacube(SRC, metadata_caching=True,
                              metadata_cache_path=str(cache), preprocess=_rename_dvar)
    assert DVAR_PP in set(cached_pp.to_xarray().data_vars)
    _assert_cube_values_match(cached_pp, open_datacube(SRC, preprocess=_rename_dvar))


@pytest.mark.integration
def test_datacube_cache_preprocess_transparency(tmp_path):
    # Multi-file: cached+preprocess == non-cached+preprocess, AND both differ from the
    # no-preprocess result (proves preprocess ran on the cache read path, which silently
    # dropped it pre-fix).
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache), concat_dims=[CDIM])

    cached_pp = open_datacube(
        files, metadata_caching=True, metadata_cache_path=str(cache),
        concat_dims=[CDIM], preprocess=_rename_dvar,
    )
    noncached_pp = open_datacube(files, concat_dims=[CDIM], preprocess=_rename_dvar)

    for c in (cached_pp, noncached_pp):
        dv = set(c.to_xarray().data_vars)
        assert DVAR_PP in dv and DVAR not in dv
    _assert_cube_values_match(cached_pp, noncached_pp)
    assert DVAR in set(open_datacube(files, concat_dims=[CDIM]).to_xarray().data_vars)


@pytest.mark.integration
def test_open_dataset_cached_applies_preprocess(tmp_path):
    # The pattern/multi-group opener routes every group through open_datacube, so
    # preprocess must flow open_dataset -> _attach_datacubes -> open_datacube on the
    # cached path too. Tag via ds.attrs (DataCube.from_xarray copies attrs -> properties).
    glob_path = os.path.join("tests", "resources",
                             "era5-single-levels-reanalysis_*.nc")
    pattern = os.path.join("tests", "resources",
                           "era5-single-levels-reanalysis_{var}.nc")
    cache = tmp_path / "ds_cache"
    build_metadata_cache(glob_path, pattern, metadata_cache_path=str(cache))

    def _tag(ds):
        ds = ds.copy()
        ds.attrs["preprocess_ran"] = "yes"
        return ds

    dset = open_dataset(glob_path, pattern, metadata_caching=True,
                        metadata_cache_path=str(cache), preprocess=_tag)
    cubes = dset.cubes  # public accessor (Dataset.cubes); eager by default
    assert len(cubes) >= 1
    for cube in cubes:
        assert cube.properties.get("preprocess_ran") == "yes"
