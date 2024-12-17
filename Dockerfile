#FROM rg.fr-par.scw.cloud/geokube/geokube-base:2024.05.03.08.14
#new version with Zarr support
FROM rg.fr-par.scw.cloud/geokube/geokube-base:2024.09.26.09.01

COPY dist/geokube-0.2.7a4-py3-none-any.whl /
RUN pip install /geokube-0.2.7a4-py3-none-any.whl --break-system-packages
RUN rm /geokube-0.2.7a4-py3-none-any.whl

#ENV LD_LIBRARY_PATH=/opt/conda/x86_64-conda-linux-gnu/lib:/usr/lib/x86_64-linux-gnu