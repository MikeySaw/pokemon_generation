# Build a GPU support docker image:
# The pytorch image needs to be pinned to this version so dvc==3.50.1 can be installed
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /wd

# Copy the package
COPY src/ src/
COPY ldm/ ldm/
COPY pokemon_stable_diffusion/ pokemon_stable_diffusion/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Install dependencies
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
# alternative to: RUN pip install -r requirements.txt --no-cache-dir

# Install the package
RUN pip install . --no-deps --no-cache-dir

# Pull the data from DVC
RUN mkdir data
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN cat .dvc/config
COPY sd-v1-4-full-ema.ckpt.dvc sd-v1-4-full-ema.ckpt.dvc
COPY *.dvc .dvc/
RUN ls -l .dvc/

RUN dvc config core.no_scm true

RUN dvc pull -v 

RUN ls -1  data/interim/train| wc -l

# Copy the necessary files for training
COPY conf/ conf/
COPY metadata.jsonl metadata.jsonl
COPY data/ data/    
RUN find data/ -type f | wc -l

# Run the training script
ENTRYPOINT ["python", "pokemon_stable_diffusion/sd_fintune.py"]
# ENTRYPOINT ["python", "-u", "pokemon_generation/modeling/train_example.py", "train"]
# run the following commands to use the file:
#    sudo docker run --gpus all -e WANDB_API_KEY=YOUR_WANDB_KEY trainer:latest
