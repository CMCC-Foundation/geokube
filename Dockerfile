#FROM rg.fr-par.scw.cloud/geokube/geokube-base:2024.05.03.08.14
#new version with Zarr support
FROM rg.fr-par.scw.cloud/geokube/geokube-base:2025.01.07.11.07

ADD . /opt/geokube
RUN pip install /opt/geokube --break-system-packages
RUN rm -rf /opt/geokube

WORKDIR /