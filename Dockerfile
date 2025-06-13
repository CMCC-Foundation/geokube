FROM rg.fr-par.scw.cloud/geokube/geokube-base:2025.06.12.14.52

ADD . /opt/geokube
RUN pip install /opt/geokube --break-system-packages
RUN rm -rf /opt/geokube

WORKDIR /