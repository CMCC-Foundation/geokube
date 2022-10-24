"""geokube framework"""
import setuptools
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path("geokube/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="geokube",
    version=main_ns["__version__"],
    author="geokube Contributors",
    author_email="geokube@googlegroups.com",
    description="Python package for Earth Science Data Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geokube/geokube",
    packages=setuptools.find_packages(),
    install_requires=[
        "bokeh==2.4.0",
        "cf_units",
        "dask",
        "distributed",
        "intake>=0.6.5",
        "xarray",
        "hvplot",
        "pytest-cov",
        "pytest",
        "shapely",
        "netCDF4",
        "scipy",
        "metpy",
        "plotly",
        "pyarrow",
        "rioxarray",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Earth Science",
    ],
    python_requires=">=3.9",
    license="Apache License, Version 2.0",
    package_data={"geokube": ["static/css/*.css", "static/html/*.html"]},
)
