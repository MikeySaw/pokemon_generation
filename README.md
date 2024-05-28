# pokemon_generation

![pokemon_generation pokemon.png](assets/pokemon.png)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

In this project we fine-tune a diffusion model on images of Pokémon. The images are annotated by labels. The goal is to have a deployable model that generates Pokémon given a text prompt.

## Experiment Command Lines Guidance(Experiments Version)
Please run `pip install -r requirements.txt` to install all the dependencies right __now__, we will use `environment.yml` file __later__. \
You need a `kaggle.json` file to activate kaggle package and its related commands, for example `kaggle --version`. \
run the following commands in command line to download zipped images from kaggle website and unzip them:
```shell
chmod +x get_images.sh
bash get_images.sh IMAGE_FOLDER.zip DESTINATION_FOLDER
```

run the following commands to test if `dvc` is working fine with your enviroment, please pin your `dvc` version to `3.50.1` so that we are using the same version not different ones. We are also going to use __Google Cloud Storage__ as our data remote storage. To do so, simply run the following commands:
```shell
# Ignore the first line if you have not installed dvc yet
pip uninstall dvc
pip install dvc==3.50.1
pip install dvc-gs

# test if the dvc is working on your PC/System
dvc pull
```


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
├── docs               <- A default mkdocs project; see mkdocs.org for details
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
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

