"""Integration tests for the kerchunk metadata cache with the reader/writer split.

The catalog (writer) builds the cache via ``build_metadata_cache``; the API (reader)
consumes it via ``open_datacube`` / ``open_dataset`` with ``metadata_caching=True``,
which never write and raise ``CacheNotExist`` if the cache is absent. Marked
``integration`` (needs the kerchunk/netCDF stack of the base image; real I/O).
"""
import math
import os

import dask
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


def _write_nc4_slab(sl, path, *, zlib=None):
    """Write a time-slice ``sl`` of the source as one NETCDF4 file.

    ``chunksizes``/``contiguous`` are dropped (avoid clashes with the slab shape). Pass
    ``zlib=True/False`` to set the data variables' compression IN PLACE (not via the
    ``encoding=`` kwarg, which would drop the grid_mapping link) — two slabs that differ
    only in compression get distinct encoding signatures and so partition apart. Used to
    construct deterministic partition layouts.
    """
    out = sl.copy()
    for v in list(out.variables):
        for enc in ("chunksizes", "contiguous"):
            out[v].encoding.pop(enc, None)
    if zlib is not None:
        for v in out.data_vars:
            out[v].encoding["zlib"] = bool(zlib)
            if zlib:
                out[v].encoding["complevel"] = 4
            else:
                out[v].encoding.pop("complevel", None)
    out.to_netcdf(path, engine="netcdf4", format="NETCDF4")
    return str(path)


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
    assert p_store["partitions"] == s_store["partitions"]  # same consolidated refs
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
        _kerchunk.load_store(str(tmp_path / "auto"))["partitions"]
        == _kerchunk.load_store(str(serial))["partitions"]
    )


@pytest.mark.integration
def test_datacube_mixed_format_combine(tmp_path):
    # Mixed NetCDF3/NetCDF4: files whose encoding signatures differ fall into separate
    # partitions (a single .zarray can't describe both layouts); each partition is
    # consolidated and the partitions are recombined at open with combine_by_coords.
    # The value-equivalence check below is the real guard.
    slabs = _make_slabs(tmp_path, formats=("nc4", "nc3"))
    mixed = slabs["nc4"][:2] + slabs["nc3"][2:]
    cache = tmp_path / "cache"
    build_metadata_cache(mixed, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) >= 1
    assert store["combine"] == "by_coords"
    cube = open_datacube(mixed, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(mixed))


@pytest.mark.integration
def test_datacube_mixed_format_unordered_combine(tmp_path):
    # Regression: mixed NetCDF3/NetCDF4 where a partition's files are supplied OUT of
    # coordinate order. Each partition is consolidated in input/file order, so its inlined
    # time axis is non-monotonic; the reader must sort EACH partition by the concat
    # coordinate before combine_by_coords -- which otherwise raises "Coordinate variable
    # time is neither monotonically increasing nor decreasing on all datasets". by_coords
    # must tolerate any file order (the prior test happened to pass files in time order).
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    n = src.sizes[CDIM]
    q = n // 4
    assert q >= 1  # SRC must have >=4 timesteps to form three equal-length slabs
    # Three EQUAL-LENGTH NetCDF4 slabs -> identical encoding signature -> ONE partition.
    nc4 = []
    for i in range(3):
        p = tmp_path / f"q{i}.nc"
        src.isel({CDIM: slice(i * q, (i + 1) * q)}).to_netcdf(
            p, engine="netcdf4", format="NETCDF4"
        )
        nc4.append(str(p))
    # One NetCDF3 slab (uncompressed/contiguous) -> different signature -> second partition.
    sl3 = src.isel({CDIM: slice(3 * q, n)}).copy()
    for v in list(sl3.variables):
        if sl3[v].dtype == np.int64:
            sl3[v] = sl3[v].astype(np.int32)
        for k in ("zlib", "complevel", "chunksizes"):
            sl3[v].encoding.pop(k, None)
    p3 = tmp_path / "q3_n3.nc"
    sl3.to_netcdf(p3, format="NETCDF3_CLASSIC")

    # nc4 partition supplied non-monotonically (down then up -> neither inc nor dec).
    mixed = [nc4[1], nc4[0], nc4[2], str(p3)]
    cache = tmp_path / "cache"
    build_metadata_cache(mixed, metadata_cache_path=str(cache))

    store = _kerchunk.load_store(str(cache))
    assert store["combine"] == "by_coords"
    assert len(store["partitions"]) >= 2  # nc4 vs nc3 signatures -> separate partitions

    cube = open_datacube(mixed, metadata_caching=True, metadata_cache_path=str(cache))
    t = cube.to_xarray()[CDIM].to_index()
    assert t.is_monotonic_increasing and len(t) == n  # full, sorted time axis
    _assert_cube_values_match(cube, open_datacube(mixed))


@pytest.mark.integration
def test_datacube_interleaved_partitions_combine(tmp_path):
    # Real-world failure: a heterogeneous archive yields SEVERAL encoding partitions whose
    # record ranges INTERLEAVE (the file encoding alternates across years). Even with each
    # partition internally monotonic, combine_by_coords cannot linearize interleaved tiles
    # ("Resulting object does not have monotonic global indexes along dimension time"); the
    # reader must concat all partitions along the record dim and sort the union. Here even
    # timesteps are zlib-compressed and odd ones uncompressed -> two distinct signatures ->
    # two partitions with interleaved times A=[t0,t2,t4], B=[t1,t3,t5]. Lazy: no data read.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    m = min(src.sizes[CDIM], 6)
    assert m >= 4
    files = []
    for i in range(m):
        sl = src.isel({CDIM: slice(i, i + 1)}).copy()
        for v in list(sl.variables):
            for k in ("chunksizes", "contiguous"):  # avoid clashes with the 1-step shape
                sl[v].encoding.pop(k, None)
        # Differentiate two encoding signatures (compressed vs not) by editing the data
        # var's encoding IN PLACE -- not via the encoding= kwarg, which would replace the
        # whole dict and drop grid_mapping (then rotated_pole would not be promoted to a
        # coordinate and would no longer match a direct open).
        sl[DVAR].encoding["zlib"] = (i % 2 == 0)
        if i % 2 == 0:
            sl[DVAR].encoding["complevel"] = 4
        else:
            sl[DVAR].encoding.pop("complevel", None)
        p = tmp_path / f"t{i}.nc"
        sl.to_netcdf(p, engine="netcdf4", format="NETCDF4")
        files.append(str(p))
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) >= 2  # compressed vs uncompressed -> interleaved parts
    cube = open_datacube(files, metadata_caching=True, metadata_cache_path=str(cache))
    t = cube.to_xarray()[CDIM].to_index()
    assert t.is_monotonic_increasing and len(t) == m  # interleaved union sorted & complete
    _assert_cube_values_match(cube, open_datacube(files))


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
def test_datacube_single_timestep_files_consolidate(tmp_path):
    # Daily-style files with ONE timestep each: xarray re-bases the time `units` per
    # file (raw value 0 in every file), so the concat coordinate must be rebuilt from
    # each file's CF-*decoded* time (not the raw manifest values) — else the time axis
    # collapses to a single step. Regression guard for the per-file decoded-time concat
    # in `_kerchunk._inline_coords`.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    n = src.sizes[CDIM]
    files = []
    for i in range(n):
        p = tmp_path / f"day{i:03d}.nc"
        src.isel({CDIM: slice(i, i + 1)}).to_netcdf(p, engine="netcdf4", format="NETCDF4")
        files.append(str(p))
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) == 1  # one signature -> consolidated into one ref
    cube = open_datacube(files, metadata_caching=True, metadata_cache_path=str(cache))
    assert cube.to_xarray().sizes[CDIM] == n  # full time axis preserved (not collapsed)
    _assert_cube_values_match(cube, open_datacube(files))


# -------------------------------------------------- byte-scalar CRS / bare record axis
# Regression for the NSIDC sea-ice CDR layout. Two traps, both reproduced here on
# synthesized files (so the suite stays independent of the large, out-of-tree
# data/NSIDC_SIC_v4 dataset):
#   1. a |S1 scalar grid-mapping container ``projection`` (value b'') -> its value is a
#      Python ``bytes`` that the store.json writer could not serialize
#      (TypeError: Object of type bytes is not JSON serializable);
#   2. a *bare* record axis ``tdim`` (the record coordinate is ``time(tdim)``, there is
#      NO ``tdim`` coordinate) -> the default by_coords open ran ``sortby("tdim")``,
#      which raises on the missing coordinate.
# The driver consumes the raw store via ``_kerchunk.open_store`` (its ``_open_main``),
# then does its own ``tdim``->``time`` swap, so these assert at the store level.

_NSIDC_CRS_ATTRS = {
    "grid_mapping_name": "polar_stereographic",
    "straight_vertical_longitude_from_pole": 135.0,
    "latitude_of_projection_origin": 90.0,
    "standard_parallel": 70.0,
    "proj4text": "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 "
                 "+a=6378273 +b=6356889.449 +units=m +no_defs",
    "srid": "urn:ogc:def:crs:EPSG::3411",
}


def _make_nsidc_like(tmp_path, times, ny=3, nx=4):
    """One single-``tdim`` NetCDF4 file per timestamp, NSIDC-style: ``siconc(tdim, y,
    x)`` with ``grid_mapping='projection'``, a non-dimension record coordinate
    ``time(tdim)``, ``xgrid(x)``/``ygrid(y)``, and a scalar |S1 ``projection``
    grid-mapping container (value b'') carrying the CRS attrs. ``siconc[i] == i``."""
    files = []
    for i, t in enumerate(times):
        ds = xr.Dataset(
            {"siconc": (("tdim", "y", "x"), np.full((1, ny, nx), float(i), "float32"))},
            coords={
                "time": ("tdim", np.array([t], dtype="datetime64[ns]")),
                "xgrid": ("x", np.arange(nx, dtype="float32")),
                "ygrid": ("y", np.arange(ny, dtype="float32")),
                "projection": np.array(b"", dtype="|S1"),
            },
        )
        ds["projection"].attrs.update(_NSIDC_CRS_ATTRS)
        ds["siconc"].attrs["grid_mapping"] = "projection"
        p = tmp_path / f"nsidc_{i:03d}.nc"
        ds.to_netcdf(p, engine="netcdf4", format="NETCDF4")
        files.append(str(p))
    return files


@pytest.mark.integration
def test_datacube_bytes_grid_mapping_scalar_roundtrips(tmp_path):
    files = _make_nsidc_like(
        tmp_path, ["1979-01-01", "1979-02-01", "1979-03-01", "1979-04-01"]
    )
    cache = tmp_path / "cache"
    # Pre-fix this raised TypeError(bytes not JSON serializable) writing store.json.
    summary = build_metadata_cache(
        files, metadata_cache_path=str(cache), combine="nested", concat_dim="tdim"
    )
    assert summary["built"] == 1 and summary["skipped"] == []
    assert os.path.isfile(cache / _kerchunk.STORE_FILE)
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) == 1

    ds = _kerchunk.open_store(store)  # exactly what the driver's _open_main consumes
    # The |S1 grid-mapping container round-tripped through the JSON sidecar...
    assert "projection" in ds.coords
    assert ds["projection"].dtype.kind == "S"
    assert ds["projection"].attrs.get("grid_mapping_name") == "polar_stereographic"
    assert ds["projection"].attrs.get("srid") == _NSIDC_CRS_ATTRS["srid"]
    # ...and the data var's grid_mapping link was restored.
    assert ds["siconc"].encoding.get("grid_mapping") == "projection"
    # Full record axis preserved (not collapsed) and values intact / in order.
    assert ds.sizes["tdim"] == len(files)
    ref = xr.open_mfdataset(
        files, combine="nested", concat_dim="tdim", decode_coords="all"
    )
    xr.testing.assert_allclose(ds["siconc"], ref["siconc"])


@pytest.mark.integration
def test_datacube_by_coords_bare_record_dim(tmp_path):
    # Default by_coords on a bare record axis: the open-time sort must order by the
    # spanning ``time`` coordinate (not the missing ``tdim`` coordinate). Files are
    # supplied OUT of time order, so the open has to sort them.
    files = _make_nsidc_like(
        tmp_path, ["1979-03-01", "1979-01-01", "1979-04-01", "1979-02-01"]
    )
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))  # default by_coords
    store = _kerchunk.load_store(str(cache))
    assert store["combine"] == "by_coords"
    ds = _kerchunk.open_store(store)  # pre-fix: raised on sortby("tdim")
    t = ds["time"].to_index()
    assert t.is_monotonic_increasing and len(t) == len(files)


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
    assert len(_kerchunk.load_store(str(cache))["partitions"]) == 1

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


# ------------------------------------- opener kwargs: lazy / graph / symmetry / sortby

def _record_var(ds):
    """A data variable that spans the record axis (the field), widest one."""
    cands = [v for v in ds.data_vars if CDIM in ds[v].dims]
    assert cands, "no data variable spans the record axis"
    return max(cands, key=lambda v: ds[v].ndim)


def _record_nblocks(ds, var):
    """Number of dask blocks along CDIM for ``var`` (asserts it is dask-backed)."""
    da = ds[var]
    assert da.chunks is not None, f"{var} is not dask-backed (eagerly loaded)"
    return len(da.chunks[da.dims.index(CDIM)])


@pytest.mark.integration
def test_cached_open_keeps_data_lazy(tmp_path):
    # The reader must NOT materialize data variables: they stay dask-backed after open.
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    cube = open_datacube(files, metadata_caching=True, metadata_cache_path=str(cache))
    xds = cube.to_xarray()
    assert xds.data_vars
    for v in xds.data_vars:
        assert dask.is_dask_collection(xds[v].data), f"{v} was eagerly loaded"


@pytest.mark.integration
def test_read_chunks_override_controls_graph(tmp_path):
    # `chunks` passed at read time reaches open_reference and controls the dask block
    # count along the record axis — the lever that bounds the task graph at scale.
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    base = _kerchunk.open_store(store)
    var, nt = _record_var(base), base.sizes[CDIM]
    assert nt > 1

    fine = _kerchunk.open_store(store, open_kwargs={"chunks": {CDIM: 1}})
    coarse = _kerchunk.open_store(store, open_kwargs={"chunks": {CDIM: -1}})
    assert _record_nblocks(fine, var) == nt   # one dask block per timestep
    assert _record_nblocks(coarse, var) == 1  # whole record axis in a single block


@pytest.mark.integration
def test_build_persists_open_kwargs_and_reader_replays(tmp_path):
    # `chunks` given at BUILD is persisted in store.json and replayed by the reader with
    # no read-time kwargs; a read-time override wins.
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache), chunks={CDIM: 2})
    store = _kerchunk.load_store(str(cache))
    assert store["open_kwargs"] == {"chunks": {CDIM: 2}}

    base = _kerchunk.open_store(store, open_kwargs={"chunks": {CDIM: 1}})
    var, nt = _record_var(base), base.sizes[CDIM]

    replayed = _kerchunk.open_store(store)  # no override -> replay persisted {CDIM: 2}
    assert _record_nblocks(replayed, var) == math.ceil(nt / 2)
    overridden = _kerchunk.open_store(store, open_kwargs={"chunks": {CDIM: 1}})
    assert _record_nblocks(overridden, var) == nt


@pytest.mark.integration
def test_sortby_skipped_when_record_coord_monotonic(tmp_path, monkeypatch):
    # In-order files -> inline record coordinate already monotonic -> no sortby graph.
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))

    calls = []
    real_sortby = xr.Dataset.sortby
    monkeypatch.setattr(
        xr.Dataset, "sortby",
        lambda self, *a, **k: (calls.append(1), real_sortby(self, *a, **k))[1],
    )
    ds = _kerchunk.open_store(store)
    assert calls == []  # monotonic -> sortby skipped
    t = np.asarray(ds[CDIM].values)
    assert np.all(t[1:] >= t[:-1])


@pytest.mark.integration
def test_sortby_runs_when_record_coord_out_of_order(tmp_path, monkeypatch):
    # WITHIN a single partition (same encoding signature) files supplied out of order are
    # consolidated in input order, so the inline record coordinate is non-monotonic and the
    # reader's sortby fallback must run. Build-time partition ordering fixes ACROSS-partition
    # disorder, not within a partition (a ManifestArray can't be reordered at build) -- so
    # this within-partition case still exercises the read-time sort.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    half = src.sizes[CDIM] // 2
    assert half >= 1
    # Two EQUAL-LENGTH nc4 slabs, same encoding -> identical signature -> ONE partition.
    lo = _write_nc4_slab(src.isel({CDIM: slice(0, half)}), tmp_path / "lo.nc")
    hi = _write_nc4_slab(src.isel({CDIM: slice(half, 2 * half)}), tmp_path / "hi.nc")
    cache = tmp_path / "cache"
    # Late slab first -> within-partition reversed -> inline coord non-monotonic.
    _kerchunk.cached_build_store([hi, lo], str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) == 1

    calls = []
    real_sortby = xr.Dataset.sortby
    monkeypatch.setattr(
        xr.Dataset, "sortby",
        lambda self, *a, **k: (calls.append(1), real_sortby(self, *a, **k))[1],
    )
    ds = _kerchunk.open_store(store)
    assert calls, "sortby must run on a non-monotonic record coordinate"
    t = np.asarray(ds[CDIM].values)
    assert np.all(t[1:] >= t[:-1])  # result ordered ascending


@pytest.mark.integration
def test_build_orders_partitions_so_reader_skips_sortby(tmp_path, monkeypatch):
    # The era5-hourly OOM shape: heterogeneous encoding splits a cube into SEVERAL
    # partitions covering DISJOINT, contiguous record ranges; supplied out of chronological
    # order, the reader's xr.concat would be non-monotonic and the open-time sortby (an
    # O(#chunks) fancy-index over the record axis -- multi-GB graph, the ~2h hang + OOM in
    # production) would fire. The build now persists partitions in record order, so the
    # concat is already chronological at block boundaries -> sortby is skipped, zero shuffle.
    # Here two contiguous halves differ only in compression (-> two signatures/partitions);
    # unlike test_datacube_interleaved_partitions_combine their ranges are DISJOINT, so
    # ordering the partition LIST fully sorts the union (no element-level sort needed).
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    L = src.sizes[CDIM]
    half = L // 2
    assert half >= 1 and L - half >= 1  # two non-empty disjoint halves
    late = _write_nc4_slab(src.isel({CDIM: slice(half, L)}), tmp_path / "late.nc", zlib=True)
    early = _write_nc4_slab(src.isel({CDIM: slice(0, half)}), tmp_path / "early.nc", zlib=False)

    cache = tmp_path / "cache"
    # Supplied LATE-first (non-chronological): first-seen partition order would interleave.
    build_metadata_cache([late, early], metadata_cache_path=str(cache))

    store = _kerchunk.load_store(str(cache))
    assert store["combine"] == "by_coords"
    assert len(store["partitions"]) == 2  # compressed vs uncompressed -> distinct signatures
    # Persisted in record order despite the late-first input: the early half (t0) is first.
    assert store["partitions"][0]["files"] == [early]
    assert store["partitions"][1]["files"] == [late]

    calls = []
    real_sortby = xr.Dataset.sortby
    monkeypatch.setattr(
        xr.Dataset, "sortby",
        lambda self, *a, **k: (calls.append(1), real_sortby(self, *a, **k))[1],
    )
    ds = _kerchunk.open_store(store)
    assert calls == []  # ordered at build -> concat already monotonic -> no sortby
    t = np.asarray(ds[CDIM].values)
    assert np.all(t[1:] >= t[:-1]) and t.size == L  # full, sorted record axis
    assert dask.is_dask_collection(ds[_record_var(ds)].data)  # data stays lazy

    cube = open_datacube([late, early], metadata_caching=True,
                         metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube([late, early]))


@pytest.mark.integration
def test_decode_kwarg_symmetry_cached_vs_direct(tmp_path):
    # A non-default decode kwarg flows to BOTH build (persisted) and read; the cached
    # cube matches a direct open with the same kwarg (cache transparency preserved).
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache), mask_and_scale=False)
    cached = open_datacube(files, metadata_caching=True,
                           metadata_cache_path=str(cache), mask_and_scale=False)
    direct = open_datacube(files, mask_and_scale=False)
    _assert_cube_values_match(cached, direct)
