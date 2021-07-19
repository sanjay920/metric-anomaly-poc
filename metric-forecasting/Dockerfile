FROM continuumio/miniconda3

SHELL ["/bin/bash", "-C"]
WORKDIR /code

# Install pre requisites
COPY ./environment.yml .
RUN conda env create -f environment.yml
RUN conda activate prophet-env

# Add code
COPY ./metric-forecasting/metric_streaming.py .

CMD python metric_streaming.py
