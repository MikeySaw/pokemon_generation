# Build a GPU support docker image with the following command:
# The pytorch image needs to be pinned to this version so dvc==3.50.1 can be installed
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc==3.50.1
RUN pip install dvc[gs]

WORKDIR /

RUN mkdir data
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN cat .dvc/config
COPY *.dvc .dvc/
RUN ls -l .dvc/

RUN dvc config core.no_scm true

RUN dvc pull -v

COPY src/  src/
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

# the following command will reuse the cache, this will make the installation process faster
# RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "pokemon_generation/modeling/train_example.py", "train"]
# run the following commands to use the file:
#    sudo docker run --gpus all -e WANDB_API_KEY=YOUR_WANDB_KEY test_trainer:latest