# load base image
FROM nvcr.io/nvidia/pytorch:24.01-py3
# without GPU: FROM python:3.11-slim

# create working directory
RUN mkdir wd
WORKDIR /wd

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy the package
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
# remove the line specifying the readme file in pyproject.toml (else the package does not install successfully)
RUN sed -ni '/readme/!p' pyproject.toml

# install dependencies
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
# alternative to: RUN pip install -r requirements.txt --no-cache-dir

# install the package
RUN pip install . --no-deps --no-cache-dir

# run the training script
ENTRYPOINT ["python", "-u", "src/modeling/OLD_training.py", "hydra.job.chdir=False"]

# run the following commands to use the file:
#    sudo docker run --gpus all -e WANDB_API_KEY=YOUR_WANDB_KEY test_trainer:latest
