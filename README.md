# geokube

## Description

geokube is a Python package based on xarray for GeoScience Data Analysis.

## Authors

**Lead Developers**:

- [Marco Mancini](https://github.com/km4rcus)
- [Jakub Walczak](https://github.com/jamesWalczak)
- [Mirko Stojiljkovic](https://github.com/MMStojiljkovic)

## Installation 

You can use pip to install `geokube`.

```bash
pip install geokube
```

#### Requirements
You need to install `xesmf` if you want to use `geokube` regridding. This can be done during the creation of conda virtual environment, as shown below

Create virtual environment with `xesmf`
```bash
conda create -n geokube python=3.11 xesmf -y
```
Activate virtual environment
```bash
conda activate geokube
```
Install geokube framework
```bash
pip install geokube
```