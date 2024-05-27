FROM python:3.11-slim
RUN pip install dvc==3.50.1
RUN pip install dvc[gs]
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
    

WORKDIR /

RUN mkdir data  
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
# show the content in .dvc/config
RUN cat .dvc/config
COPY *.dvc .dvc/
RUN ls -l .dvc/

RUN dvc config core.no_scm true

# Set up Google Cloud Storage remote with the service account

RUN dvc pull -v

# check the files inside the data folder
RUN ls -l data/