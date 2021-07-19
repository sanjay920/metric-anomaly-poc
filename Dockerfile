FROM continuumio/miniconda3

WORKDIR /code

# Install pre requisites
COPY ./environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "prophet-env", "/bin/bash", "-c"]

# Add code
COPY ./metric-forecasting/metric_streaming.py .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "prophet-env", "python", "./metric_streaming.py"]
