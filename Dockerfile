FROM continuumio/miniconda3:22.11.1
COPY environment.yaml /tmp/enviroment.yaml
RUN conda config --set allow_conda_downgrades true 
RUN conda env update --name base --file /tmp/enviroment.yaml --prune
RUN conda install -c conda-forge gxx_linux-64==11.1.0
RUN conda update -n base -c defaults conda
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete
COPY dist/geokube-0.2.6b0-py3-none-any.whl /
RUN pip install /geokube-0.2a0-py3-none-any.whl
RUN rm /geokube-0.2.6b0-py3-none-any.whl
ENV LD_LIBRARY_PATH=/opt/conda/x86_64-conda-linux-gnu/lib:/usr/lib/x86_64-linux-gnu