# Use the same base image as your training Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the necessary files for inference
COPY conf/ /app/conf/
COPY ldm/ /app/ldm/
COPY pokemon_stable_diffusion/ /app/pokemon_stable_diffusion/
COPY requirements.txt /app/requirements.txt
COPY sd-v1-4-full-ema.ckpt /app/sd-v1-4-full-ema.ckpt

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install fastapi uvicorn

# Copy the FastAPI app
COPY src/app.py /app/app.py

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
