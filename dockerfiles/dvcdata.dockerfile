FROM python:3.11-slim
RUN pip install dvc==3.42.0
RUN pip install dvc[gs]
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN mkdir data  
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN echo "DVC config:" && cat .dvc/config || echo "DVC config not found"
COPY *.dvc .dvc/
RUN echo "DVC files:" && ls -la *.dvc || echo "No DVC files found"

RUN dvc config core.no_scm true

# Set up Google Cloud Storage remote with the service account

RUN dvc pull -v || true

RUN echo "Contents of data directory:" && ls -la data/ || echo "Data directory is empty"
RUN dvc status || true
RUN pwd
