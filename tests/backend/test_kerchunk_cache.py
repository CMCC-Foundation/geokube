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
from geokube.core.axis import AxisType
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


def _write_chunked_slab(sl, path, *, time_chunk):
    """Write a time-slice as NETCDF4 with an explicit on-disk record chunk of ``time_chunk``.

    Stale chunk/contiguous encoding is dropped first, then the record-spanning data vars are
    chunked at ``time_chunk`` along ``CDIM`` (full extent on the other axes). Choosing a
    ``time_chunk`` that does not divide the slab's record length makes the file carry a
    partial final chunk along the record axis — the era5-hourly-extended shape VirtualiZarr
    cannot consolidate via a ManifestArray concat. Compression is dropped so slabs written
    this way share one encoding signature (they would consolidate into one partition but for
    the irregular grid).
    """
    out = sl.copy()
    for v in list(out.variables):
        # ``original_shape``/``preferred_chunks`` must go too: xarray's netcdf4 backend
        # ignores a forced ``chunksizes`` while ``original_shape`` still reflects the source
        # (a shape-mismatch guard), silently keeping the source chunk (here time=1).
        for enc in ("chunksizes", "contiguous", "original_shape", "preferred_chunks"):
            out[v].encoding.pop(enc, None)
    for v in out.data_vars:
        if CDIM in out[v].dims:
            out[v].encoding["chunksizes"] = tuple(
                time_chunk if d == CDIM else out.sizes[d] for d in out[v].dims
            )
            out[v].encoding["contiguous"] = False
            out[v].encoding.pop("zlib", None)
            out[v].encoding.pop("complevel", None)
    out.to_netcdf(path, engine="netcdf4", format="NETCDF4")
    return str(path)


def _two_var_src():
    """Source with TWO data variables sharing dims/coords (DVAR + a derived second var),
    to exercise the single-variable-per-file -> per-variable-partition case."""
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    src = src.assign({"VAR2": src[DVAR] + 1.0})
    gm = src[DVAR].encoding.get("grid_mapping") or src[DVAR].attrs.get("grid_mapping")
    if gm:
        src["VAR2"].encoding["grid_mapping"] = gm
    return src


def _write_var_file(src, var, tsl, path, *, zlib=None):
    """Write a single-variable NETCDF4 file: only ``var`` (+ its coords) over time ``tsl``."""
    return _write_nc4_slab(src[[var]].isel({CDIM: tsl}), path, zlib=zlib)


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


@pytest.mark.integration
def test_open_dataset_lazy_cubes_expose_variables(tmp_path):
    # Regression for the SPS3.5 many-cube catalog: with delay_read_cubes=True the cubes
    # are Delayed, so geokube must load the variable list (and to_dict / ds[var]) from ONE
    # representative cube (groups are homogeneous) instead of returning `datacube: None`
    # or an empty ds[var]. Two time-slabs of the same source = a 2-group homogeneous
    # catalog (same variable, distinct record ranges), keyed by a non-variable attribute.
    from dask.delayed import Delayed

    _make_slabs(tmp_path, n=2)  # -> slab0.nc, slab1.nc (same var, different time range)
    glob_path = os.path.join(str(tmp_path), "slab*.nc")
    pattern = os.path.join(str(tmp_path), "slab{tag}.nc")
    cache = tmp_path / "cache"
    summary = build_metadata_cache(glob_path, pattern, metadata_cache_path=str(cache))
    assert summary["groups"] == 2 and summary["built"] == 2

    # eager open defines the expected schema. Field names are geokube's (keyed by
    # standard_name, e.g. `air_temperature`), distinct from the netCDF var name (ncvar).
    expected_vars = set(
        open_dataset(glob_path, pattern, metadata_caching=True,
                     metadata_cache_path=str(cache)).variables
    )
    assert expected_vars  # non-empty: the representative cube resolved the schema
    field = next(iter(expected_vars))

    # lazy open: cubes stay Delayed, but the variable list is still exposed.
    dset = open_dataset(glob_path, pattern, metadata_caching=True,
                        metadata_cache_path=str(cache), delay_read_cubes=True)
    assert all(isinstance(c, Delayed) for c in dset.cubes)
    assert set(dset.variables) == expected_vars
    # reading the schema must NOT materialize the stored cubes (only a cached representative).
    assert all(isinstance(c, Delayed) for c in dset.cubes)

    # to_dict is populated from the representative (fields + domain), not None.
    entries = dset.to_dict()
    assert len(entries) == 2
    for e in entries:
        assert e["datacube"] is not None
        assert set(e["datacube"]["fields"]) == expected_vars
        assert "domain" in e["datacube"]

    # selection stays lazy, by field name AND by the netCDF var name (ncvar, how the API
    # queries); an unknown variable -> empty Dataset.
    for sel in (field, DVAR):
        sub = dset[sel]
        assert len(sub) == 2 and all(isinstance(c, Delayed) for c in sub.cubes)
    assert len(dset["does_not_exist"]) == 0


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
    # Partitions keep first-seen (input) order: late-first -> p0=late, p1=early.
    assert store["partitions"][0]["files"] == [late]
    assert store["partitions"][1]["files"] == [early]
    # Distinct record axes -> two singleton merge groups, ordered by record start despite
    # the late-first input (early half=p1 listed first) -> reader concats chronologically.
    assert store["combine_plan"] == [[1], [0]]

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
def test_single_variable_files_merge_not_concat(tmp_path):
    # The era5-hourly bug: SINGLE-variable files -> one encoding partition PER VARIABLE,
    # all sharing the same record axis. A blind xr.concat along time would STACK them ->
    # time axis inflated N-fold with each variable on only its slice (wrong + the OOM
    # driver). The combine_plan groups same-axis partitions into ONE merge group, so the
    # reader MERGES the variables onto the single shared time axis instead.
    src = _two_var_src()
    L = src.sizes[CDIM]
    f1 = _write_var_file(src, DVAR, slice(None), tmp_path / "v1.nc")
    f2 = _write_var_file(src, "VAR2", slice(None), tmp_path / "v2.nc")

    cache = tmp_path / "cache"
    build_metadata_cache([f1, f2], metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) == 2          # one partition per variable
    assert store["combine_plan"] == [[0, 1]]       # same record axis -> one merge group

    ds = _kerchunk.open_store(store)
    assert int(ds.sizes[CDIM]) == L                # MERGED: time NOT inflated to 2*L
    assert set(ds.data_vars) >= {DVAR, "VAR2"}     # both variables present on one axis
    assert dask.is_dask_collection(ds[DVAR].data)  # lazy

    cube = open_datacube([f1, f2], metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube([f1, f2]))


@pytest.mark.integration
def test_single_variable_two_tiles_merge_and_concat(tmp_path):
    # Full era5 shape: 2 variables x 2 disjoint time tiles, each (var,tile) a single-variable
    # file whose tiles differ in compression (-> 4 distinct signatures/partitions). The plan
    # must MERGE the two variables within each tile (same axis) and CONCAT the two tiles
    # (distinct axes) along time -> 2 vars over the full, un-inflated record axis.
    src = _two_var_src()
    L = src.sizes[CDIM]
    h = L // 2
    assert h >= 1 and L - h >= 1
    # tile A (t0..h-1) compressed, tile B (h..L-1) uncompressed; supplied tile-B first.
    bv1 = _write_var_file(src, DVAR, slice(h, L), tmp_path / "b_v1.nc", zlib=False)
    bv2 = _write_var_file(src, "VAR2", slice(h, L), tmp_path / "b_v2.nc", zlib=False)
    av1 = _write_var_file(src, DVAR, slice(0, h), tmp_path / "a_v1.nc", zlib=True)
    av2 = _write_var_file(src, "VAR2", slice(0, h), tmp_path / "a_v2.nc", zlib=True)
    files = [bv1, bv2, av1, av2]

    cache = tmp_path / "cache"
    build_metadata_cache(files, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert len(store["partitions"]) == 4
    plan = store["combine_plan"]
    assert len(plan) == 2 and all(len(g) == 2 for g in plan)  # 2 tiles, 2 vars merged each

    ds = _kerchunk.open_store(store)
    assert int(ds.sizes[CDIM]) == L                # full axis: tiles concatenated, not stacked
    assert set(ds.data_vars) >= {DVAR, "VAR2"}     # both variables merged across tiles
    t = np.asarray(ds[CDIM].values)
    assert np.all(t[1:] >= t[:-1])                 # chronological
    assert dask.is_dask_collection(ds[DVAR].data)

    cube = open_datacube(files, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(files))


@pytest.mark.integration
def test_irregular_chunk_grid_splits_to_per_file(tmp_path, monkeypatch):
    # The era5-hourly-extended crash: SAME-encoding files whose record-axis length is NOT a
    # multiple of their on-disk record chunk (production: 8760 timesteps chunked at 512). A
    # build-time ManifestArray concat needs a regular chunk grid, so VirtualiZarr rejects the
    # partial mid-array chunk ("Cannot concatenate arrays with partial chunks"). The build must
    # instead emit ONE partition PER FILE and let the (dask-backed) reader concatenate them.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    L = src.sizes[CDIM]
    half = L // 2
    if half < 3:
        pytest.skip("source record axis too short to exercise a partial mid-array chunk")
    # Two equal, disjoint halves with identical encoding -> normally ONE consolidated
    # partition. A record chunk of half-1 does not divide half (half >= 3), so each file has a
    # partial final chunk -> irregular grid -> the non-last input is rejected by a build concat.
    chunk_t = half - 1
    a = _write_chunked_slab(src.isel({CDIM: slice(0, half)}), tmp_path / "a.nc", time_chunk=chunk_t)
    b = _write_chunked_slab(src.isel({CDIM: slice(half, 2 * half)}), tmp_path / "b.nc", time_chunk=chunk_t)

    cache = tmp_path / "cache"
    # Supplied late-first (b, a) to also exercise chronological reordering at read.
    summary = build_metadata_cache([b, a], metadata_cache_path=str(cache))
    assert summary["built"] == 1 and summary["skipped"] == []  # build no longer crashes

    store = _kerchunk.load_store(str(cache))
    # Same signature would consolidate to 1 partition; the irregular grid forces per-file.
    assert len(store["partitions"]) == 2
    assert [p["files"] for p in store["partitions"]] == [[b], [a]]  # first-seen (input) order
    # Two disjoint singleton groups, ordered by record start -> a (early, idx 1) before b.
    assert store["combine_plan"] == [[1], [0]]

    calls = []
    real_sortby = xr.Dataset.sortby
    monkeypatch.setattr(
        xr.Dataset, "sortby",
        lambda self, *a, **k: (calls.append(1), real_sortby(self, *a, **k))[1],
    )
    ds = _kerchunk.open_store(store)
    assert calls == []  # groups ordered at build -> concat already chronological -> no sortby
    t = np.asarray(ds[CDIM].values)
    assert t.size == 2 * half and np.all(t[1:] >= t[:-1])  # full, chronological record axis
    assert dask.is_dask_collection(ds[_record_var(ds)].data)  # data stays lazy

    cube = open_datacube([b, a], metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube([b, a]))


@pytest.mark.integration
def test_nested_irregular_chunk_grid_stays_lazy(tmp_path, monkeypatch):
    # era5 nested variant: combine="nested" + an irregular chunk grid -> per-file partitions.
    # The reader must LAZILY xr.concat them in input order, NOT eager xr.combine_nested, whose
    # default compat computes overlapping vars aligned across the disjoint time ranges (the
    # 48 GiB OOM in staging). Regression guard for the nested multi-partition read path.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    L = src.sizes[CDIM]
    half = L // 2
    if half < 3:
        pytest.skip("source record axis too short to exercise a partial mid-array chunk")
    chunk_t = half - 1
    a = _write_chunked_slab(src.isel({CDIM: slice(0, half)}), tmp_path / "a.nc", time_chunk=chunk_t)
    b = _write_chunked_slab(src.isel({CDIM: slice(half, 2 * half)}), tmp_path / "b.nc", time_chunk=chunk_t)

    cache = tmp_path / "cache"
    build_metadata_cache([a, b], metadata_cache_path=str(cache), combine="nested", concat_dim=CDIM)
    store = _kerchunk.load_store(str(cache))
    assert store["combine"] == "nested"
    assert len(store["partitions"]) == 2  # irregular grid -> per-file, not consolidated

    def _boom(*a, **k):
        raise AssertionError("reader used eager xr.combine_nested (the OOM path)")

    monkeypatch.setattr(xr, "combine_nested", _boom)
    ds = _kerchunk.open_store(store)
    assert int(ds.sizes[CDIM]) == 2 * half                    # lazily concatenated in input order
    assert dask.is_dask_collection(ds[_record_var(ds)].data)  # data stays lazy (no compute)
    monkeypatch.undo()

    cube = open_datacube([a, b], metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube([a, b]))


@pytest.mark.integration
def test_nested_no_concat_dim_irregular_grid_stays_lazy(tmp_path, monkeypatch):
    # The exact staging shape: combine="nested" built WITHOUT an explicit concat_dim -> the
    # store persists concat_dim=None. With an irregular chunk grid -> per-file partitions, the
    # reader must RESOLVE the concat dim from the data and lazily xr.concat, never fall back to
    # eager xr.combine_nested (which computes across the disjoint ranges -> the 48 GiB OOM).
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    L = src.sizes[CDIM]
    half = L // 2
    if half < 3:
        pytest.skip("source record axis too short to exercise a partial mid-array chunk")
    chunk_t = half - 1
    a = _write_chunked_slab(src.isel({CDIM: slice(0, half)}), tmp_path / "a.nc", time_chunk=chunk_t)
    b = _write_chunked_slab(src.isel({CDIM: slice(half, 2 * half)}), tmp_path / "b.nc", time_chunk=chunk_t)

    cache = tmp_path / "cache"
    build_metadata_cache([a, b], metadata_cache_path=str(cache), combine="nested")  # no concat_dim
    store = _kerchunk.load_store(str(cache))
    assert store["combine"] == "nested" and store["concat_dim"] is None  # nothing persisted
    assert len(store["partitions"]) == 2  # irregular grid -> per-file

    def _boom(*a, **k):
        raise AssertionError("reader used eager xr.combine_nested (the OOM path)")

    monkeypatch.setattr(xr, "combine_nested", _boom)
    ds = _kerchunk.open_store(store)                           # concat dim resolved at read
    assert int(ds.sizes[CDIM]) == 2 * half                    # lazily concatenated, full axis
    assert dask.is_dask_collection(ds[_record_var(ds)].data)  # data stays lazy (no compute)
    monkeypatch.undo()

    cube = open_datacube([a, b], metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube([a, b]))


@pytest.mark.integration
def test_undecodable_time_units_honors_decode_times_false(tmp_path):
    # sps3p5 EQM shape: monthly data whose time is non-CF ("months since 1993-01-01" with a
    # proleptic_gregorian calendar, which xarray/cftime cannot decode -> only 360_day allows
    # "months since"). The per-file manifest build (reference_one) must honor decode_times=False
    # forwarded from open_kwargs -- like the non-cached open_mfdataset path -- so it does not
    # crash; geokube's cf_units layer interprets the unit downstream. Pre-fix reference_one
    # ignored open_kwargs, so it decoded (default) and crashed even with decode_times=False.
    n = 6
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), np.zeros((n, 3, 4), dtype="float32"))},
        coords={"time": ("time", np.arange(n, dtype="int32")),
                "y": ("y", np.arange(3)), "x": ("x", np.arange(4))},
    )
    ds["time"].attrs = {
        "units": "months since 1993-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "standard_name": "time",
        "axis": "T",
    }
    p = str(tmp_path / "monthly.nc")
    ds.to_netcdf(p, engine="netcdf4", format="NETCDF4")

    cache = str(tmp_path / "cache")
    summary = build_metadata_cache(p, metadata_cache_path=cache, decode_times=False)
    assert summary["built"] == 1 and summary["skipped"] == []  # build did not crash

    ds2 = _kerchunk.open_store(_kerchunk.load_store(cache), open_kwargs={"decode_times": False})
    t = np.asarray(ds2["time"].values)
    assert not np.issubdtype(t.dtype, np.datetime64)          # kept raw (not xarray-decoded)
    assert t.size == n and int(t[0]) == 0 and int(t[-1]) == n - 1
    assert str(ds2["time"].attrs.get("units", "")).startswith("months since")  # unit preserved


def _write_monthly(path, *, n=6, ref="1993-01-01 00:00:00",
                   calendar="proleptic_gregorian", start=0):
    """Write a monthly NetCDF4 file whose time is a non-CF 'months since ...' reference."""
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), np.zeros((n, 3, 4), dtype="float32"))},
        coords={"time": ("time", np.arange(start, start + n, dtype="int32")),
                "y": ("y", np.arange(3)), "x": ("x", np.arange(4))},
    )
    ds["time"].attrs = {"units": f"months since {ref}", "calendar": calendar,
                        "standard_name": "time", "axis": "T"}
    ds.to_netcdf(str(path), engine="netcdf4", format="NETCDF4")
    return str(path)


def _expected_month_starts(n=6, year=1993):
    return np.array([f"{year}-{m:02d}-01" for m in range(1, n + 1)], dtype="datetime64[ns]")


def test_decode_month_year_reference_routine():
    # Pure decode routine: months/years offsets -> calendar-correct datetime64, incl. the
    # ref.day>28 month-end clamping fallback. No I/O -> runs in the fast (non-integration) suite.
    from geokube.core.variable import _decode_month_year_reference

    m = _decode_month_year_reference(
        np.arange(6), "months since 1993-01-01 00:00:00", "proleptic_gregorian"
    )
    assert np.array_equal(m.astype("datetime64[ns]"), _expected_month_starts())

    y = _decode_month_year_reference(np.arange(3), "years since 2000-01-01", "standard")
    assert list(y.astype("datetime64[Y]").astype(str)) == ["2000", "2001", "2002"]

    # ref.day = 31 -> +1 month must clamp to Feb 29 (2020 is a leap year), not overflow.
    c = _decode_month_year_reference(
        np.array([0, 1]), "months since 2020-01-31 00:00:00", "standard"
    )
    assert str(c[0]).startswith("2020-01-31") and str(c[1]).startswith("2020-02-29")

    # a decodable "days since" unit is left to xarray -> routine returns None.
    assert _decode_month_year_reference(np.arange(3), "days since 1970-01-01", None) is None


@pytest.mark.integration
def test_undecodable_time_build_without_flag_decodes_end_to_end(tmp_path):
    # The build AUTO-detects the non-CF "months since" unit, forces+persists decode_times=False
    # (the DDS driver need pass nothing), and open_datacube decodes months -> datetime64.
    p = _write_monthly(tmp_path / "monthly.nc")
    cache = str(tmp_path / "cache")

    summary = build_metadata_cache(p, metadata_cache_path=cache)  # NO decode_times passed
    assert summary["built"] == 1 and summary["skipped"] == []  # auto-detect prevented the crash
    assert _kerchunk.load_store(cache)["open_kwargs"].get("decode_times") is False  # persisted

    cube = open_datacube(p, metadata_caching=True, metadata_cache_path=cache)
    tvals = np.asarray(cube.to_xarray()[CDIM].values)
    assert np.issubdtype(tvals.dtype, np.datetime64)
    assert np.array_equal(tvals.astype("datetime64[ns]"), _expected_month_starts())


@pytest.mark.integration
def test_undecodable_time_non_cached_decodes_same(tmp_path):
    # Cached and non-cached opens share the Variable.from_xarray funnel, so both decode
    # identically (the non-cached path needs the explicit decode_times=False to open raw).
    p = _write_monthly(tmp_path / "monthly.nc")
    cube = open_datacube(p, decode_times=False)
    tvals = np.asarray(cube.to_xarray()[CDIM].values)
    assert np.issubdtype(tvals.dtype, np.datetime64)
    assert np.array_equal(tvals.astype("datetime64[ns]"), _expected_month_starts())


@pytest.mark.integration
def test_undecodable_time_to_dict_min_max_step(tmp_path):
    # to_dict reports correct monthly dates/step, not the pre-fix garbage epoch dates.
    p = _write_monthly(tmp_path / "monthly.nc")
    cache = str(tmp_path / "cache")
    build_metadata_cache(p, metadata_cache_path=cache)
    cube = open_datacube(p, metadata_caching=True, metadata_cache_path=cache)

    tcoord = cube.domain[AxisType.TIME]
    d = tcoord.to_dict()
    assert str(d["min"]).startswith("1993-01-01")
    assert str(d["max"]).startswith("1993-06-01")
    assert d["time_unit"] == "month" and d["time_step"] == 1
    assert str(d["units"]).startswith("months since")


def test_is_undecodable_time_unit_predicate():
    # The predicate that drives both the build sniff and the read decode. No I/O.
    from geokube.utils.attrs_encoding import is_undecodable_time_unit

    assert is_undecodable_time_unit("months since 1993-01-01 00:00:00", "proleptic_gregorian")
    assert is_undecodable_time_unit("years since 2000-01-01", "standard")
    # 360_day CAN decode months/years via cftime -> NOT undecodable.
    assert not is_undecodable_time_unit("months since 1993-01-01", "360_day")
    # standard sub-monthly units decode natively, incl. the AxisType.TIME default.
    assert not is_undecodable_time_unit("days since 1970-01-01", None)
    assert not is_undecodable_time_unit("hours since 1970-01-01", "gregorian")
    assert not is_undecodable_time_unit("seconds since 2000-01-01", "proleptic_gregorian")
    assert not is_undecodable_time_unit("kelvin", None)
    assert not is_undecodable_time_unit(None, None)


@pytest.mark.integration
def test_time_units_undecodable_probe(tmp_path):
    # The build-time metadata probe detects the non-CF month unit but not a decodable one.
    assert _kerchunk._time_units_undecodable(_write_monthly(tmp_path / "m.nc"))
    ds = xr.Dataset({"v": (("time",), np.zeros(3, "float32"))},
                    coords={"time": ("time", np.arange(3, dtype="int32"))})
    ds["time"].attrs = {"units": "days since 1970-01-01", "calendar": "standard"}
    reg = str(tmp_path / "days.nc")
    ds.to_netcdf(reg, engine="netcdf4", format="NETCDF4")
    assert not _kerchunk._time_units_undecodable(reg)


@pytest.mark.integration
def test_undecodable_time_to_xarray_roundtrip(tmp_path):
    # A decoded month axis must round-trip through to_xarray/to_netcdf without hitting the
    # cftime "months since" encode limitation (units normalized to a CF-decodable encoding).
    p = _write_monthly(tmp_path / "monthly.nc")
    cache = str(tmp_path / "cache")
    build_metadata_cache(p, metadata_cache_path=cache)
    cube = open_datacube(p, metadata_caching=True, metadata_cache_path=cache)

    out = str(tmp_path / "roundtrip.nc")
    cube.to_xarray().to_netcdf(out)  # must not raise
    back = xr.open_dataset(out)
    assert np.array_equal(
        np.asarray(back[CDIM].values).astype("datetime64[ns]"), _expected_month_starts()
    )


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


def _write_broken_time_bnds(path):
    """A file whose ``time`` is a decodable ``days since`` unit but whose ``time_bnds``
    carries an int32 fill sentinel (``-2147483648``) with NO units of its own.

    Mirrors the soil-erosion-vhr-era5 ``historical`` files: because ``time`` declares
    ``bounds='time_bnds'``, xarray's CF decode inherits ``time``'s units/calendar onto
    ``time_bnds`` and tries to decode ``-2147483648 days`` -> OverflowError -> the
    ``ValueError: unable to decode time units 'days since 1991-01-01'`` seen in prod.
    ``decode_times=False`` is the only correct handling (the value is undecodable garbage).
    """
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), np.zeros((1, 3, 4), dtype="float32")),
         "time_bnds": (("time", "bnds"),
                       np.array([[0.0, -2147483648.0]], dtype="float64"))},
        coords={"time": ("time", np.array([5297.0], dtype="float64")),
                "y": ("y", np.arange(3)), "x": ("x", np.arange(4))},
    )
    ds["time"].attrs = {"units": "days since 1991-01-01", "calendar": "proleptic_gregorian",
                        "standard_name": "time", "axis": "T", "bounds": "time_bnds"}
    ds["time"].encoding = {"_FillValue": None}
    ds["time_bnds"].encoding = {"_FillValue": None}  # keep the sentinel unmasked on disk
    ds.to_netcdf(str(path), engine="netcdf4", format="NETCDF4")
    return str(path)


@pytest.mark.integration
def test_broken_time_bnds_transparent_with_decode_times_false(tmp_path):
    # With decode_times=False (as the catalog config sets, and the driver now forwards to the
    # build) the manifest build does not crash, the flag is persisted, and the cached read
    # matches the direct non-cached open -- transparent. Time stays raw; time_bnds preserved.
    p = _write_broken_time_bnds(tmp_path / "hist.nc")
    cache = str(tmp_path / "cache")

    summary = build_metadata_cache(p, metadata_cache_path=cache, decode_times=False)
    assert summary["built"] == 1 and summary["skipped"] == []
    assert _kerchunk.load_store(cache)["open_kwargs"].get("decode_times") is False  # persisted

    cached = open_datacube(p, metadata_caching=True, metadata_cache_path=cache,
                           decode_times=False)
    direct = open_datacube(p, decode_times=False)
    _assert_cube_values_match(cached, direct)

    xc, xd = cached.to_xarray(), direct.to_xarray()
    # Time stays a raw axis (matches the configured non-cached decode_times=False open;
    # geokube only self-decodes months/years, not days-since) and is identical to the
    # direct open. Data-variable transparency is covered by _assert_cube_values_match above.
    assert not np.issubdtype(np.asarray(xc[CDIM].values).dtype, np.datetime64)
    np.testing.assert_array_equal(
        np.asarray(xc[CDIM].values), np.asarray(xd[CDIM].values)
    )


@pytest.mark.integration
def test_broken_time_bnds_default_build_raises(tmp_path):
    # Documents WHY the flag is required: a single-cube build with default decoding surfaces
    # the time-decode failure (single-cube builds are not silently swallowed; only the
    # multi-cube catalog loop isolates a bad cube, see the isolation test below).
    p = _write_broken_time_bnds(tmp_path / "hist.nc")
    cache = str(tmp_path / "cache")
    with pytest.raises(ValueError, match="decode time"):
        build_metadata_cache(p, metadata_cache_path=cache)


@pytest.mark.integration
def test_failing_cube_isolated_in_pattern_build(tmp_path):
    # A multi-cube (pattern) build must isolate an unrecoverable cube: report it in `skipped`
    # and still build the healthy cubes, instead of aborting the whole run. Here the bad cube
    # is the broken-time_bnds file opened with default decoding (proactive probe cannot see a
    # value overflow); the healthy cube is a "months since" file the probe auto-handles.
    good = _write_monthly(tmp_path / "good.nc")
    bad = _write_broken_time_bnds(tmp_path / "bad.nc")
    cache = str(tmp_path / "cache")
    pattern = str(tmp_path / "{var}.nc")

    summary = build_metadata_cache(  # must NOT raise
        str(tmp_path / "*.nc"), pattern=pattern, metadata_cache_path=cache,
    )
    assert summary["groups"] == 2
    assert summary["built"] == 1
    assert len(summary["skipped"]) == 1 and "bad" in repr(summary["skipped"])
    _ = good  # (referenced for clarity; the healthy cube is the one that built)


def _write_optional_bnds(path, *, var, with_bnds, n=3):
    """A small NetCDF4 file with data variable ``var`` over a ``time`` axis, optionally
    carrying a ``time_bnds`` bounds variable — to exercise ``drop_variables`` over files
    where the dropped var is present in only some of them (the bioclimind shape)."""
    dv = {var: (("time", "y", "x"), np.zeros((n, 3, 4), dtype="float32"))}
    if with_bnds:
        dv["time_bnds"] = (("time", "bnds"),
                           np.stack([np.arange(n), np.arange(1, n + 1)], axis=1).astype("float64"))
    ds = xr.Dataset(dv, coords={"time": ("time", np.arange(n, dtype="int32")),
                                "y": ("y", np.arange(3)), "x": ("x", np.arange(4))})
    ds["time"].attrs = {"units": "days since 1970-01-01", "calendar": "standard",
                        "standard_name": "time", "axis": "T"}
    if with_bnds:
        ds["time"].attrs["bounds"] = "time_bnds"
    ds.to_netcdf(str(path), engine="netcdf4", format="NETCDF4")
    return str(path)


@pytest.mark.integration
def test_drop_variables_lenient_when_absent(tmp_path):
    # Regression: `drop_variables` naming a var ABSENT from the file must not crash the build.
    # VirtualiZarr's open_virtual_dataset(drop_variables=...) uses a strict drop_vars that raises
    # on a missing var; geokube applies it leniently (errors="ignore") in reference_one instead.
    p = _write_optional_bnds(tmp_path / "nobnds.nc", var="v", with_bnds=False)
    cache = str(tmp_path / "cache")
    summary = build_metadata_cache(p, metadata_cache_path=cache, drop_variables=["time_bnds"])
    assert summary["built"] == 1 and summary["skipped"] == []  # no strict-drop crash


@pytest.mark.integration
def test_drop_variables_lenient_across_mixed_files(tmp_path):
    # bioclimind shape: a pattern build where the dropped var (time_bnds) is present in only
    # SOME files. Every cube must build (lenient drop where present, ignore where absent) —
    # none skipped, no raise.
    _write_optional_bnds(tmp_path / "aaa.nc", var="aaa", with_bnds=True)
    _write_optional_bnds(tmp_path / "bbb.nc", var="bbb", with_bnds=False)
    cache = str(tmp_path / "cache")
    pattern = str(tmp_path / "{var}.nc")
    summary = build_metadata_cache(
        str(tmp_path / "*.nc"), pattern=pattern, metadata_cache_path=cache,
        drop_variables=["time_bnds"],
    )
    assert summary["groups"] == 2
    assert summary["built"] == 2 and summary["skipped"] == []


# ------------------------------------------------------ passthrough: contiguous vars
# A variable whose native (on-disk) chunk spans the whole array (contiguous/unchunked
# HDF5/NetCDF4 storage, or any NetCDF3-classic non-record variable) is excluded from
# the kerchunk reference entirely and reopened directly from source at read time
# instead of being cached as one indivisible chunk that no `chunks=` could subdivide.

def _write_contiguous_slab(sl, path):
    """Write a time-slice as NETCDF4 with data variables EXPLICITLY contiguous
    (unchunked), regardless of any ambient chunking default -- the on-disk layout
    kerchunk represents as one indivisible reference chunk (see
    ``_kerchunk._is_contiguous_chunk``). HDF5 forbids contiguous storage for a
    variable with an unlimited dimension, so ``unlimited_dims=[]`` is forced too."""
    out = sl.copy()
    for v in list(out.variables):
        for enc in ("chunksizes", "contiguous", "original_shape", "preferred_chunks"):
            out[v].encoding.pop(enc, None)
    for v in out.data_vars:
        out[v].encoding["contiguous"] = True
        out[v].encoding.pop("zlib", None)
        out[v].encoding.pop("complevel", None)
    out.to_netcdf(path, engine="netcdf4", format="NETCDF4", unlimited_dims=[])
    return str(path)


def _write_contiguous_nc3(sl, path):
    """Write a time-slice as NETCDF3_CLASSIC. The format has no chunking concept for
    non-record variables at all, so as long as no dim is unlimited every data
    variable is contiguous by construction (``kerchunk.netCDF3.NetCDF3ToZarr``
    reports ``chunks=shape`` for it)."""
    out = sl.copy()
    for v in list(out.variables):
        if out[v].dtype == np.int64:
            out[v] = out[v].astype(np.int32)
        for k in ("zlib", "complevel", "chunksizes", "contiguous"):
            out[v].encoding.pop(k, None)
    out.to_netcdf(path, format="NETCDF3_CLASSIC", unlimited_dims=[])
    return str(path)


def _write_record_nc3(path, n_time=2500):
    """Synthetic NetCDF3_CLASSIC file with ``time`` as the RECORD (unlimited)
    dimension -- kerchunk references such a variable one reference chunk PER RECORD
    (``kerchunk.netCDF3.NetCDF3ToZarr.translate``), the mirror-image pathology to a
    contiguous (whole-array) chunk. Used to exercise ``_warn_if_many_small_chunks``."""
    ds = xr.Dataset(
        {"v": (("time", "y"), np.arange(n_time * 2, dtype="float32").reshape(n_time, 2))},
        coords={"time": ("time", np.arange(n_time, dtype="int32"))},
    )
    ds.to_netcdf(path, format="NETCDF3_CLASSIC", unlimited_dims=["time"])
    return str(path)


@pytest.mark.integration
def test_passthrough_splices_small_chunks_for_contiguous_variable(tmp_path):
    # A contiguous variable must NOT end up as one giant dask block: it is excluded
    # from the kerchunk reference and reopened directly from source instead.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    p = _write_contiguous_slab(src, tmp_path / "contig.nc")
    cache = tmp_path / "cache"
    build_metadata_cache(p, metadata_cache_path=str(cache))

    store = _kerchunk.load_store(str(cache))
    assert store["partitions"][0].get("passthrough"), "no variable was flagged passthrough"
    assert DVAR in store["partitions"][0]["passthrough"]
    assert store["partitions"][0]["passthrough"][DVAR]["engine"] == "h5netcdf"

    # The manifest itself must not carry a reference for the passthrough variable.
    ref = os.path.join(str(cache), store["partitions"][0]["parquet"])
    assert DVAR not in _kerchunk.open_reference(ref).data_vars

    cube = open_datacube(p, metadata_caching=True, metadata_cache_path=str(cache))
    xds = cube.to_xarray()
    assert dask.is_dask_collection(xds[DVAR].data)
    _assert_cube_values_match(cube, open_datacube(p))


@pytest.mark.integration
def test_passthrough_lets_auto_chunk_subdivide_reference_cannot(tmp_path):
    # The core value proposition: `chunks="auto"` (the bare-default lever for
    # passthrough -- see `_kerchunk._passthrough_chunks`) against the REFERENCE path
    # is constrained by dask to a multiple of the on-disk chunk declared in the
    # manifest -- for a contiguous variable that IS the whole array, so auto can only
    # ever pick ONE block, no matter how small the byte-size target. The native
    # engine passthrough uses exposes no preferred chunk for a contiguous variable at
    # all, so auto is free to size chunks from the byte-size target alone -- shrink
    # that target so even this small fixture visibly subdivides.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    p = _write_contiguous_slab(src, tmp_path / "contig.nc")

    with dask.config.set({"array.chunk-size": "256KiB"}):
        cache_on = tmp_path / "cache_on"
        build_metadata_cache(p, metadata_cache_path=str(cache_on))
        on = _kerchunk.open_store(_kerchunk.load_store(str(cache_on)))
        assert _record_nblocks(on, DVAR) > 1

        cache_off = tmp_path / "cache_off"
        build_metadata_cache(
            p, metadata_cache_path=str(cache_off), passthrough_contiguous=False,
        )
        off = _kerchunk.open_store(_kerchunk.load_store(str(cache_off)))
        assert _record_nblocks(off, DVAR) == 1  # reference path: stuck at one chunk


@pytest.mark.integration
def test_passthrough_disabled_restores_single_chunk_reference(tmp_path):
    # `passthrough_contiguous=False` restores the pre-existing (single indivisible
    # chunk) behavior and never populates the `passthrough` store field.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    p = _write_contiguous_slab(src, tmp_path / "contig.nc")
    cache = tmp_path / "cache"
    build_metadata_cache(p, metadata_cache_path=str(cache), passthrough_contiguous=False)

    store = _kerchunk.load_store(str(cache))
    assert not store["partitions"][0].get("passthrough")
    cube = open_datacube(p, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(p))


@pytest.mark.integration
def test_passthrough_netcdf3_nonrecord_uses_netcdf4_engine(tmp_path):
    # A contiguous NetCDF3-classic (non-record) variable is ALSO a passthrough
    # candidate, but must be reopened with the `netcdf4` engine -- `h5netcdf` cannot
    # even open a NetCDF3 file.
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    p = _write_contiguous_nc3(src, tmp_path / "contig_n3.nc")
    cache = tmp_path / "cache"
    build_metadata_cache(p, metadata_cache_path=str(cache))

    store = _kerchunk.load_store(str(cache))
    passthrough = store["partitions"][0].get("passthrough") or {}
    assert DVAR in passthrough
    assert passthrough[DVAR]["engine"] == "netcdf4"

    cube = open_datacube(p, metadata_caching=True, metadata_cache_path=str(cache))
    _assert_cube_values_match(cube, open_datacube(p))


@pytest.mark.integration
def test_passthrough_max_files_cap_falls_back(tmp_path, caplog):
    # A contiguous variable spanning more files than the cap falls back to today's
    # single-chunk reference for it instead of reopening that many files natively.
    # Four EQUAL-length slabs -> identical shape/encoding signature -> one partition
    # (an uneven split would give each its own signature and defeat the cap check).
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    n = src.sizes[CDIM]
    q = n // 4
    assert q >= 1, "SRC must have >=4 timesteps to form four equal-length slabs"
    contiguous = [
        _write_contiguous_slab(
            src.isel({CDIM: slice(i * q, (i + 1) * q)}), tmp_path / f"c{i}.nc"
        )
        for i in range(4)
    ]
    cache = tmp_path / "cache"
    with caplog.at_level("WARNING", logger="_kerchunk.py"):
        build_metadata_cache(
            contiguous, metadata_cache_path=str(cache),
            passthrough_max_files_per_partition=2,
        )
    assert "cap 2" in caplog.text

    store = _kerchunk.load_store(str(cache))
    assert not any(p.get("passthrough") for p in store["partitions"])
    cube = open_datacube(
        contiguous, metadata_caching=True, metadata_cache_path=str(cache)
    )
    _assert_cube_values_match(cube, open_datacube(contiguous))


@pytest.mark.integration
def test_passthrough_ignores_already_chunked_variable(tmp_path):
    # Regression: a genuinely chunked (non-contiguous) variable must never be marked
    # passthrough -- only the reference path applies to it.
    slabs = _make_slabs(tmp_path)["nc4"]
    src = xr.open_dataset(slabs[0], decode_coords="all")
    src.load()
    p = _write_chunked_slab(src, tmp_path / "chunked.nc", time_chunk=1)
    cache = tmp_path / "cache"
    build_metadata_cache(p, metadata_cache_path=str(cache))
    store = _kerchunk.load_store(str(cache))
    assert not store["partitions"][0].get("passthrough")


@pytest.mark.integration
def test_passthrough_config_change_invalidates_cache(tmp_path):
    # `passthrough_contiguous`/`passthrough_max_files_per_partition` are part of the
    # manifest context: flipping them must force a rebuild (not an incremental no-op).
    src = xr.open_dataset(SRC, decode_coords="all")
    src.load()
    p = _write_contiguous_slab(src, tmp_path / "contig.nc")
    cache = tmp_path / "cache"
    build_metadata_cache(p, metadata_cache_path=str(cache))
    assert _kerchunk.load_store(str(cache))["partitions"][0].get("passthrough")

    build_metadata_cache(p, metadata_cache_path=str(cache), passthrough_contiguous=False)
    assert not _kerchunk.load_store(str(cache))["partitions"][0].get("passthrough")


@pytest.mark.integration
def test_many_small_chunks_warns_for_netcdf3_record_variable(tmp_path, caplog):
    # A NetCDF3 record variable is referenced one kerchunk chunk PER RECORD -- the
    # opposite pathology to a contiguous variable. Opening it with the bare default
    # `chunks={}` would build one dask task per record; warn at build time instead.
    p = _write_record_nc3(tmp_path / "record.nc", n_time=2500)
    cache = tmp_path / "cache"
    with caplog.at_level("WARNING", logger="_kerchunk.py"):
        build_metadata_cache(p, metadata_cache_path=str(cache))
    assert "native reference chunks" in caplog.text


@pytest.mark.integration
def test_many_small_chunks_no_warning_with_explicit_chunks(tmp_path, caplog):
    # The warning is only useful when the caller left `chunks` at the bare default;
    # an explicit coalescing `chunks=` (or "auto") silences it.
    p = _write_record_nc3(tmp_path / "record.nc", n_time=2500)
    cache = tmp_path / "cache"
    with caplog.at_level("WARNING", logger="_kerchunk.py"):
        build_metadata_cache(p, metadata_cache_path=str(cache), chunks={"time": 500})
    assert "native reference chunks" not in caplog.text


# ------------------------------------------------------------- build: progress bar
# `progress=True` shows nested tqdm bars while building. It must be a pure add-on:
# identical summary + identical store as `progress=False`, and it must actually drive
# tqdm (we spy on the loader). Off by default -> tqdm is never even loaded.

class _FakeTqdm:
    """Records instantiations; works both as an iterable wrapper (``progress_iter``)
    and as a total/update/close bar (the dask callback). Crucially it re-yields the
    wrapped iterable unchanged, so the build sees every file."""

    calls = 0

    def __init__(self, iterable=None, **kwargs):
        type(self).calls += 1
        self._iterable = iterable

    def __iter__(self):
        return iter(() if self._iterable is None else self._iterable)

    def update(self, n=1):
        pass

    def close(self):
        pass


@pytest.fixture
def spy_tqdm(monkeypatch):
    from geokube.backend import _progress

    _FakeTqdm.calls = 0
    monkeypatch.setattr(_progress, "_load_tqdm", lambda: _FakeTqdm)
    return _FakeTqdm


@pytest.mark.integration
def test_build_progress_is_transparent_single_cube(tmp_path, spy_tqdm):
    files = _make_slabs(tmp_path)["nc4"]
    plain, bar = tmp_path / "plain", tmp_path / "bar"
    s_plain = build_metadata_cache(files, metadata_cache_path=str(plain))
    assert spy_tqdm.calls == 0  # progress defaults off -> tqdm never loaded/used
    s_bar = build_metadata_cache(files, metadata_cache_path=str(bar), progress=True)
    assert spy_tqdm.calls > 0  # the per-file bar(s) were driven
    assert s_bar == s_plain
    assert (
        _kerchunk.load_store(str(bar))["partitions"]
        == _kerchunk.load_store(str(plain))["partitions"]
    )
    _assert_cube_values_match(
        open_datacube(files, metadata_caching=True, metadata_cache_path=str(bar)),
        open_datacube(files),
    )


@pytest.mark.integration
def test_build_progress_with_dask_scheduler(tmp_path, spy_tqdm):
    # The dask path drives the bar via a dask.diagnostics callback (local scheduler).
    # "synchronous" exercises the delayed/compute plumbing deterministically.
    files = _make_slabs(tmp_path)["nc4"]
    cache = tmp_path / "cache"
    summary = build_metadata_cache(
        files, metadata_cache_path=str(cache), scheduler="synchronous", progress=True
    )
    assert summary["built"] == 1
    assert spy_tqdm.calls > 0
    _assert_cube_values_match(
        open_datacube(files, metadata_caching=True, metadata_cache_path=str(cache)),
        open_datacube(files),
    )


@pytest.mark.integration
def test_build_progress_pattern_outer_cube_bar(tmp_path, spy_tqdm):
    glob_path = os.path.join("tests", "resources",
                             "era5-single-levels-reanalysis_*.nc")
    pattern = os.path.join("tests", "resources",
                           "era5-single-levels-reanalysis_{var}.nc")
    plain, bar = tmp_path / "plain", tmp_path / "bar"
    s_plain = build_metadata_cache(glob_path, pattern, metadata_cache_path=str(plain))
    s_bar = build_metadata_cache(
        glob_path, pattern, metadata_cache_path=str(bar), progress=True
    )
    assert s_bar == s_plain
    assert spy_tqdm.calls > 0  # outer cube bar + inner per-file bar(s)
    assert open_dataset(glob_path, pattern, metadata_caching=True,
                        metadata_cache_path=str(bar)) is not None
