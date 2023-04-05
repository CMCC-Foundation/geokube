# geokube

## Description

geokube is a Python package for analysis of Earth Science Data.

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

#### Conda Package
Create virtual environment with installing conda-build package
```bash
conda create -n geokube python=3.9 conda-build -y
```
Activate virtual environment
```bash
conda activate geokube
```
Execute the script
```bash
./conda/build_and_install.sh
```