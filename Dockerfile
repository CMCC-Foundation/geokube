#FROM rg.fr-par.scw.cloud/geokube/geokube-base:2024.05.03.08.14
#new version with Zarr support
FROM rg.fr-par.scw.cloud/geokube/geokube-base:2025.01.09.14.31

ADD . /opt/geokube
RUN pip install /opt/geokube --break-system-packages
RUN rm -rf /opt/geokube

WORKDIR /