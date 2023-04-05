FROM continuumio/miniconda3:22.11.1
RUN conda config --set allow_conda_downgrades true 
RUN conda install -c conda-forge --yes --freeze-installed cartopy 
RUN conda install -c conda-forge --yes --freeze-installed 'bokeh>=2.4.2,<3'
RUN conda install -c conda-forge --yes --freeze-installed numpy=1.23.5
RUN conda install -c conda-forge --yes --freeze-installed pandas=1.4.3
RUN conda install -c conda-forge --yes --freeze-installed netCDF4
RUN conda install -c conda-forge --yes --freeze-installed scipy 
RUN conda install -c conda-forge --yes --freeze-installed xarray=2022.6.0
RUN conda install -c conda-forge --yes --freeze-installed xesmf=0.6.3
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete
COPY dist/geokube-0.2a0-py3-none-any.whl /
RUN pip install /geokube-0.2a0-py3-none-any.whl
RUN rm /geokube-0.2a0-py3-none-any.whl
