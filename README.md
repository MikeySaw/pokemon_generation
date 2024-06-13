# pokemon_generation

![pokemon_generation pokemon.png](assets/pokemon.png)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

In this project we fine-tune a diffusion model on images of Pokémon. The images are annotated by labels. The goal is to have a deployable model that generates Pokémon given a text prompt.

## Experiment Command Lines Guidance(Experiments Version)
### Data Download Part
Please run `pip install -r requirements.txt` to install all the dependencies right __now__, we will use `environment.yml` file __later__. \
You need a `kaggle.json` file to activate kaggle package and its related commands, for example `kaggle --version`. \
run the following commands in command line to download zipped images from kaggle website and unzip them:
```shell
chmod +x get_images.sh
bash get_images.sh IMAGE_FOLDER.zip DESTINATION_FOLDER
```
### Data Version Control Test
run the following commands to test if `dvc` is working fine with your enviroment, please pin your `dvc` version to `3.50.1` so that we are using the same version not different ones. We are also going to use __Google Cloud Storage__ as our data remote storage. To do so, simply run the following commands:
```shell
# Ignore the first line if you have not installed dvc yet
pip uninstall dvc
pip install dvc==3.50.1
pip install dvc-gs

# test if the dvc is working on your PC/System
dvc pull
```

### Hydra Test
please check the `src/config` folder for different hyperparameter settings, right now the files inside the folder are all __placeholder__, which means that the real config conresponding values are not fitted inside the folder yet, to add your own experiment hyperparameters, simply add another `yaml` file inside the `src/config/experiments` folder, please beware of the required formats of the hyperparameter yaml files, you need to add this \
```shell
# @package _global_
``` 
at the beginning of your yaml files so that later we can directly change the config files we gonna use from command line like this way: 
```shell
# change the default hyperparameter values tom values inside the train_1.yaml file
python train.py config=train_1.yaml
```
The structure of this folder should always looks similar to this one: 
```shell
├── config
├── default_config.yaml
└── experiments
    ├── train_1.yaml
    └── train_2.yaml
```

### Dockerfile Test
please read the `test_trainer.dockfile` for more details, this file is used to be a showcase for building everything, aka `dvc`&`CUDA`&`ENTRYPOINT` in one dockerfile. 
to make this dockerfile easier to understand, a toy example is added to the `src/model/train_example.py`, this is the entrypoint of the dockerfile.
to build and test this toy example dockerfile, simply run the following command:
```shell
# build dockerfile
sudo docker build -f test_trainer.dockerfile . -t test_trainer:latest

# test dockerfile
sudo docker run --gpus all -e WANDB_API_KEY=YOUR_WANDB_KEY test_trainer:latest
```
__make sure to replace the `YOUR_WANDB_KEY` here with your real wandb personel token!__

### Model Building and Multi-GPUs training with Diffusers and Huggingface
To train a diffusion model with Multi-GPUs and algorithm like `LoRA`, please run the following commands:
```shell
pip install accelerate
pip install git+https://github.com/huggingface/diffusers
# Instead you can install by using pip install -r requirements.txt

# Initialize the config for the accelerate
accelerate config

# Train the model by using the following script
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="models/finetune/lora"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="data/raw"

accelerate launch --mixed_precision="fp16"  notebooks/train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_NAME \
--dataloader_num_workers=8 \
--resolution=224 \
--center_crop \
--random_flip \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--max_train_steps=15000 \
--learning_rate=1e-04 \
--max_grad_norm=1 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir=${OUTPUT_DIR} \
--push_to_hub \
--hub_model_id=${HUB_MODEL_ID} \
--report_to=wandb \
--checkpointing_steps=500 \
--validation_prompt="A naruto with blue eyes." \
--multi_gpu 2 \
--seed=42
```
Our trainining would be done on two `A6000` GPUs with 40GB RAM for each of them. 

### Run model training in a docker container
To run the model training script src/modeling/training.py in a reproducible docker container first build an image using the following command:
```console
docker build -f dockerfiles/training.dockerfile . -t training:<image_tag>
```
Then run the training script in a container using:
```console
docker run --name <container_name> --rm \
    -v $(pwd)/data:/wd/data                             `# mount the data folder` \
    -v $(pwd)/models:/wd/models                         `# mount the model folder` \
    -v $(pwd)/hydra_logs/training_outputs:/wd/outputs   `# mount the hydra logging folder` \
    training:<image_tag> \
    paths.model_name=model0 \
    paths.training_data=data/processed/pokemon.pth
```

### Workspace cleaning and garbage collection
To remove a docker image run the following:
```console
docker rmi <image_name>:<image_tag>
```
To run docker garbage collection run the following:
```console
docker system prune -f
```
To delete all unused images (warning) and run docker garbage collection run the following:
```console
docker system prune -af
```
TODO: add a "make clean" command to the Makefile

### Dataset Structure
Right now the `data` folder is not uploaded to 🤗 Datasets, we may consider to upload this folder to the 🤗 Datasets if we use a dataset with JSON file as meta info at the end of this project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── dockerfiles        <- Dockerfiles for reproducible training and inference.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── hydra_logs         <- Logging information on training and inference runs of models.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for src
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── training.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

