FROM continuumio/miniconda3:22.11.1
COPY environment.yaml /tmp/enviroment.yaml
RUN conda config --set allow_conda_downgrades true 
RUN conda env update --name base --file /tmp/enviroment.yaml --prune
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete
COPY dist/geokube-0.3.0a0-py3-none-any.whl /
RUN pip install /geokube-0.3.0a0-py3-none-any.whl
RUN rm /geokube-0.3.0a0-py3-none-any.whl