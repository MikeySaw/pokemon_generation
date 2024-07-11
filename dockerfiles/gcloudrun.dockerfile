# Build a GPU support docker image:
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

# Copy the necessary files for inference
COPY conf/ conf/
COPY sd-v1-4-full-ema.ckpt sd-v1-4-full-ema.ckpt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI app
CMD ["uvicorn", "src/app:app", "--host", "0.0.0.0", "--port", "8080"]
