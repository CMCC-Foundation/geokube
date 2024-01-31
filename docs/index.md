# Welcome to geokube

**geokube** is an open source Python package for geoscience data analysis that provides the user with a simple application programming interface (API) for performing geospatial operations (e.g., extracting a bounding box or regridding) and temporal operations (e.g., resampling) on different types of scientific feature types like grids, profiles and points, using on `xarray` data structures like `xarray.Dataset` and custom `xarray.Dataset.indexes` and xarray ecosystem frameworks such as `xesmf` and `cf-xarray`.

Furthermore, based on xarray IO engines, geokube provides built-in drivers for reading geoscientific datasets produced by Earth Observations (e.g. satellite data) and Earth Systems Models (e.g., weather, climate and oceanographic models)