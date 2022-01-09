# GeoKube

#### Prerequisites
You need to install xesmf and cartopy to use geokube. This can be done during the creation of conda virtual environment, as shown below

Add or append conda-forge channel
```bash
conda config --add channels conda-forge
```
or
```bash
conda config --append channels conda-forge
```

#### GeoKube installation
Create virtual environment with installing xesmf and cartopy package
```bash
conda create -n geokube python=3.8 xesmf=0.6.1 cartopy=0.18 -y
```
Activate virtual environment
```bash
conda activate geokube
```
Install geokube framework
```bash
python setup.py insttall
```

#### Conda package creation
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