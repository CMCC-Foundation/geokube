# Production image: installs the geokube package on top of the prebuilt base
# (ESMF / esmpy / xesmf stack). The base reference is injected at build time so
# the image is reproducible — CI pins it to the exact digest of
# geokube-base:latest; locally it defaults to :latest.
ARG BASE_IMAGE=rg.fr-par.scw.cloud/geokube/geokube-base:latest
FROM ${BASE_IMAGE}

# The base puts its virtualenv on PATH, so pip targets the venv directly
# (no --break-system-packages needed).
COPY . /opt/geokube
RUN pip install --no-cache-dir /opt/geokube \
 && rm -rf /opt/geokube

WORKDIR /
