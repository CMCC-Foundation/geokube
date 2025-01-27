# geokube

## Description

**geokube** is an open source Python package for geoscience data analysis that provides the user with a simple application programming interface (API) for performing geospatial operations (e.g., extracting a bounding box or regridding) and temporal operations (e.g., resampling) on different types of scientific feature types like grids, profiles and points, using  `xarray` data structures and xarray ecosystem frameworks such as `xesmf`.

## Developers Team

- [Valentina Scardigno](https://github.com/vale95-eng)
- [Gabriele Tramonte](https://github.com/gtramonte)

### Former Developers

- [Marco Mancini](https://github.com/km4rcus)
- [Jakub Walczak](https://github.com/jamesWalczak)
- [Mirko Stojiljkovic](https://github.com/MMStojiljkovic)

## Installation 

#### Requirements
You need to install xesmf and cartopy to use some feature of geokube.

```bash
pip install geokube==v0.2.7.2
```

#### Docker Image
Prebuilt Docker images of Geokube are available:

```bash
docker pull rg.fr-par.scw.cloud/geokube/geokube:v0.2.7.2
```