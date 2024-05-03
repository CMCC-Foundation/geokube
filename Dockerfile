FROM rg.fr-par.scw.cloud/geokube/geokube-base:2024.05.03.08.14

COPY dist/geokube-0.2.6b2-py3-none-any.whl /
RUN pip install /geokube-0.2.6b2-py3-none-any.whl
RUN rm /geokube-0.2.6b2-py3-none-any.whl

#ENV LD_LIBRARY_PATH=/opt/conda/x86_64-conda-linux-gnu/lib:/usr/lib/x86_64-linux-gnu