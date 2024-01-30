# geokube

## Description

**geokube** is an open source Python package for geoscience data analysis that provides the user with a simple application programming interface (API) for performing geospatial operations (e.g., extracting a bounding box or regridding) and temporal operations (e.g., resampling) on different types of scientific feature types like grids, profiles and points, using  `xarray` data structures and xarray ecosystem frameworks such as `xesmf`.

## Authors

**Lead Developers**:

- [Marco Mancini](https://github.com/km4rcus)
- [Jakub Walczak](https://github.com/jamesWalczak)
- [Mirko Stojiljkovic](https://github.com/MMStojiljkovic)

## Installation 

#### Requirements
You need to install xesmf and cartopy to use geokube. This can be done during the creation of conda virtual environment, as shown below

Add or append conda-forge channel
```bash
conda config --add channels conda-forge
```
or
```bash
conda config --append channels conda-forge
```

#### Conda Environment
Create virtual environment with installing xesmf and cartopy package
```bash
conda create -n geokube python=3.9 xesmf=0.6.2 cartopy -y
```
Activate virtual environment
```bash
conda activate geokube
```
Install geokube framework
```bash
python setup.py install
```