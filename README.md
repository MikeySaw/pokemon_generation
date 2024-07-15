# pokemon_generation

![pokemon_generation pokemon.png](assets/pokemon.png)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

In this project we fine-tune a diffusion model on images of Pokémon. The images are annotated by labels. The goal is to have a deployable model that generates Pokémon given a text prompt.

<a name="top"></a>
![Finetuned_pokemon.png](assets/sample_1.png)

## Acknowledgement

Everyone contributed equal and faily during the whole project! 🙌🙌🙌

## Contribution Guidance
Our project has been open-sourced now, if you want to contribute to our project, please follow the following instructions. Have fun coding!

### Starting Point Alarm! 🚨 <a href="#top">[Back to Top]</a>

Before start to `git add` anything related to this repo, please make sure you run the following commands! 

```shell
# Get the newest version of the repo！
git pull origin main

# install the newest version dependencies!
pip install -r real_requirements.txt

# run the pre-commit hook to check/modify your file you wanna push!
pip install pre-commit 

# Alert!!!💥 The following line will check every files in the repo based on the pre-commit hook!
pre-commit run --all-files

# Only want to check one file?  Use this command instead!
pre-commit run --files YOUR_FILE_NAME

# Then do the normal procedure 💯
# git add / git commit / git push ...
```
__Please always open a pull request if you want to merge your modification to the repo!__ 🤗

## Table of Contents

- [Model Training](#Model-training-)
  - [Finetune a Stable Diffusion](#Finetune-a-stable-diffusion-)
  - [Test Stable Diffusion Model with a dummy input](#Test-stable-diffusion-model-with-a-dummy-input-)
- [Model Preparation](#Model-preparation-)
  - [Data Download Part](#Data-download-part-)
  - [Data Version Control](#Data-version-control-%EF%B8%8F)
  - [Reproduce Dataset creation](#Reproduce-dataset-creation-%EF%B8%8F)
  - [Hydra Test](#Hydra-test-)
  - [Github Actions & Continuous Integration & Docker Build Workflow](#Github-actions--continuous-integration--docker-build-workflow-)
  - [Pre-Commit Hook](#Pre-commit-hook-%EF%B8%8F)
  - [Pytest Test](#Pytest-test-%EF%B8%8F)
  - [Coverage](#Coverage)
  - [Dockerfile Test](#Dockerfile-test-)
  - [Dockerfile Building Up commands](#Dockerfile-building-up-commmands-)
- [Cloud Training commands](#Cloud-training-commands-%EF%B8%8F)
- [Deploy Model Via FastAPI](#Deploy-model-via-fastapi-)
- [Serve Model Locally](#Serve-model-locally-)
- [Deploy model via Google Cloud](#Deploy-model-via-google-cloud)
  - [Deploy model via Google Cloud Function](#Deploy-model-via-google-cloud-function-)
  - [Deploy model via Google Cloud Run](#Deploy-model-via-google-cloud-run-)
- [Data Shifting Check](#Data-shifting-check)
- [Pytorch Lightning Training, Profiling, DDP and Distributed Data Loading](#Pytorch-lightning-training-profiling-ddp-and-distributed-data-loading-%EF%B8%8F)
- [Model Pruning&Compiling&Quantization](#Model-pruningcompilingquantization-)
- [Run model training locally](#Run-model-training-locally)
- [Run model training in a docker container](#Run-model-training-in-a-docker-container)
- [Workspace cleaning and garbage collection](#Workspace-cleaning-and-garbage-collection)

## Model Training 🌋 <a href="#top">[Back to Top]</a> <a name="model-training"></a>
_TL, DR. I just want to train my model!_ 🤘
### Finetune a Stable Diffusion 🔥 <a href="#top">[Back to Top]</a> <a name="Finetune-a-stable-diffusion"></a>
To finetune a Stable Diffusion Model simply run the following commands:
```shell
# Get the repo!
git clone https://github.com/MikeySaw/pokemon_generation/
cd pokemon_generation/

# Get the env!
conda create -n pokemon python==3.11
conda activte pokemon
pip install -r real_requirements.txt

# Get the data and the origin model weights!
dvc pull

# Train the model!
python pokemon_stable_diffusion/sd_finetune.py
```
To train a `DDPM` model from scratch(Backbone of our SD), simply run the following commands.
```shell
cd src/modeling/
python train_ddpm_example.py
```

__Alert!!!🚨 You must have a very nice GPU if you want to run the training commands!__

### Test Stable Diffusion Model with a dummy input <a href="#top">[Back to Top]</a> <a name="Test-stable-diffusion-model-with-a-dummy-input-"></a>

To test Stable Diffusion Model with a dummy input (already prepared for you!), simply run the following commands:

```
python pokemon_stable_diffusion/latent_diffusion.py 
```

This will run the `dummy training` process based on a `dummy image` and a `dummy txt`. \
You will see the generated images `sample_0.png`, if the code is executed correctly.
__Alert!🚨 You need to work on a very expensive server if you want to test this code!(at least 24GB RAM)__

### Installaton Debug Guidance
You may encounter issues when you install `requirements.txt` via command line. This would be caused by those following lines inside the `requirements.txt` file:
```shell
-e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
-e git+https://github.com/openai/CLIP.git@main#egg=clip
```
You may need to manually install those two packages if the issues persist.

## Model Preparation 🛸 <a name="Model Preparation-"></a>

### Data Download Part 🚚 <a href="#top">[Back to Top]</a> <a name="Data Download Part-"></a>

Please run `pip install -r requirements.txt` to install all the dependencies right __now__, we will use `environment.yml` file __later__. \
You need a `kaggle.json` file to activate kaggle package and its related commands, for example `kaggle --version`. \
run the following commands in command line to download zipped images from kaggle website and unzip them:

```shell
chmod +x get_images.sh
bash get_images.sh IMAGE_FOLDER.zip DESTINATION_FOLDER
```

### Data Version Control ⚙️ <a href="#top">[Back to Top]</a> <a name="Data-version-control-%EF%B8%8F"></a>

run the following commands to test if `dvc` is working fine with your enviroment, please pin your `dvc` version to `3.50.1` so that we are using the same version not different ones. This will avoid version conflict problems during the dockerfile building phase. We are also going to use __Google Cloud Storage__ as our data remote storage. To do so, simply run the following commands:

```shell
# Ignore the first line if you have not installed dvc yet
pip uninstall dvc
pip install dvc==3.50.1
pip install dvc-gs

# test if the dvc is working on your PC/System
dvc pull
```

### Reproduce Dataset creation 🖼️ <a href="#top">[Back to Top]</a> <a name="Reproduce-dataset-creation-%EF%B8%8F"></a>

If you want to create a dataset with your own images run the following. This will generate captions for your images, move the images and created jsonl files to their respective train/test/val folders and create a dataset for you. Make sure to have your images in the `data/raw` directory:

```shell
# generate captions with BLIP2
python src/data/add_data_description.py
# create train/test/val split
python src/data/create_data_splits.py
# create a torch dataset for train/test/val split
python src/data/make_dataset.py
```

### Hydra Test 👾 <a href="#top">[Back to Top]</a> <a name="Hydra-test-"></a>

please check the `config/` folder for different hyperparameter settings, to add your own experiment hyperparameters, simply add another `yaml` file inside the `config/` folder, please beware of the required formats of the hyperparameter yaml files, you need to add this \

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

We can change the config settings during the _training/sampling_ in command line, it would be something like this:

```shell
python train.py optimizer=sgd
```

### Github Actions & Continuous Integration & Docker Build Workflow 🐝 <a href="#top">[Back to Top]</a> <a name="Github-actions--continuous-integration--docker-build-workflow-"></a>

For github actions related file, please check the `.github/workflows`, this folder includes all the github actions which will be trigged when we push/pull into our repo, to be more specific about those files, here is a brief introduction about what those files are doing: \
the `ci.yaml` file would be responsible for `continuous integration` operation, trigger this github action file will trigger the `tests` folder and all the `pytest` files inside this repo.
the `lint.yaml` file would be responsible for `pre-commit` hook, this hook will check all the formats we want to use for our files inside this repo.
When pull/merge to the github repo, the Google Cloud will automatically trigger the docker image build workflow, the `cloudbuild.yaml` dockerfile will build a dockerfile for testing `dvc pull` command for getting the data.

#### Pre-Commit Hook 🕵️ <a href="#top">[Back to Top]</a> <a name="Pre-commit-hook-%EF%B8%8F"></a> 

To check the detailed configs about the `pre-commit` hook, please check the `.pre-commit-config.yaml` file. If you are not satisfied with the style we are using, simply change settings inside this file!

### Pytest Test ✔️ <a href="#top">[Back to Top]</a> <a name="Pytest-test-%EF%B8%8F"></a>

To run `.py` files related to the  `pytest` package, simply run the following command:

```shell
pytest tests/
```

this will run all the files inside the `tests` folder named as `tests_ ...`

Wanna add your own `pytest` check into the repo? Easy! Simply add a `.py` file inside the `tests` folder, the file should be named as `test_...`, then add libraries and functions inside this file, the function should also be written like:

```shell
def test_...(*args, **kwargs):
    ...
```

### Coverage ⌛ <a href="#top">[Back to Top]</a> <a name="coverage"></a>
To calculate the coverage rate of all the `pytest` related tests, simply run the following commands:
```shell
coverage run -m pytest tests/

# check the coverage report!
coverage report
``` 

### Dockerfile Test 🐋 <a href="#top">[Back to Top]</a> <a name="dockerfile test"></a>

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

### Dockerfile Building Up commmands 🐳 <a href="#top">[Back to Top]</a> <a name="Dockerfile Building Up commmands"></a>
To build the training dockerfile, please run the following commands:
```shell
# If you encounter issues, consider use `sudo` before the whole command
docker build -f sd_finetune.dockerfile . -t fd_train:latest
```

To overwrite an entrypoint of the dockerfile, simply run the following commands:
```shell
docker run --gpus all -e WANDB_API_KEY=YOUR_WANDB_TOKEN --entrypoint python fd_train:latest pokemon_stable_diffusion/sd_finetune.py
```

For `MAC A1/A2` chip user, you may consider to use this command if you want to deploy the model on cloud later:
```shell
docker build --platform linux/amd64 -f sd_finetune.dockerfile . -t fd_train:latest
```

To build the data test dockerfile to test if `dvc` is working correctly, simply run the following codes:
```shell
# If you encounter issues, consider use `sudo` before the whole command
docker build -f dvcdata.dockerfile . -t fd_data:latest
```

To build upon `app.py` and deploy your lovely model on _Google Cloud_ later, simply run the following commands:
```shell
docker build -f gcloudrun.dockerfile . -t gcp_test_app:latest
```

To run the training dockerfile you just build, simply run the following commands: \
__Alert! 🚨The following dockerfile includes GPU training support, automatical dvc data preparation, and Wandb logging, please make sure you have all the env prepared!__
__Alert! 🚨The Stable Diffusion fine-tuning needs at least 18 GB RAM GPU to run, use server or consider rent a GPU if you want to run the following dockerfile__
```shell
docker run --gpus all -e WANDB_API_KEY=YOUR_WANDB_KEY fd_train:latest
```
Please replace the `YOUR_WANDB_KEY` with your own `wandb` authorization token, to get your own token, simply click the following link: [wandb authorization link](https://wandb.ai/authorize), then login and copy paste your own authorization token.
Please do not forget the `--gpus all` flags, this will automatically🪄activate your _NVIDIA GPU_ if your machine has one. Enjoy the fast training! 🏄‍♀️

#### Docker Debug Guidance🧑🏿‍🔧
Before you start to build another (large!) dockerfile, you may consider to check which dockerfile you already have:
```shell
docker images
```
If you find out you accidently built a dockerfile you do not need anymore, run the following command to delete the dockerfile
```shell
docker rmi IMAGE_ID
```
If you encounter issues with deleting the dockerfiles, copy paste the sequence of numbers at the end of your error message, then try the following two commands:
```shell
docker rm numbers
# or
docker rmi numbers
# then try to delete the docker images again
docker rmi IMAGE_ID
```

If `--gpus all` flag returns an error with GPU support, you may need to check the following commands:
```shell
# check if the nvidia-driver is installed 
# go to their website and download the driver if you do not have one already
nvidia-smi

# check if the compiler is correct/cuda tookit is available
nvcc --version
# you may need sudo rights if nvcc command is not recognized by your machine
# sudo apt install nvidia-cuda-toolkit
```

If the commands before did not solve the error you are encountering, you may need an extra tookit for your dockerfile to run with a GPU support;
```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
After running those commands, your dockerfile should now work with GPU support very smoothly!🏎️

### Cloud Training commands ☁️ <a href="#top">[Back to Top]</a> <a name="Cloud Training commands"></a>
To start the cloud training in _GCloud Compute Engine_ with _Nvidia GPU_ support, simply run the following commands to check the available GPUs in different ZONE first:
```shell
gcloud compute accelerator-types list
```
Since we are not going to train the whole model on _GCloud Compute Engine_ engine, we do not need anything more advanced than _Nvidia T4_, also, it is really hard and expensive to get any GPU besides the _T4_.
Try to run the following command to see if we could successfully create a compute engine with GPU support:
```shell
gcloud compute instances create adios1 \
--zone="asia-northeast3-c" \
--image-family="pytorch-latest-gpu" \
--image-project=deeplearning-platform-release \
--accelerator="type=nvidia-tesla-t4,count=1" \
--maintenance-policy TERMINATE 
```
When you successfully created an instance, _ssh_ to the instance to launch your training.
```shell
# check the compute instances we created already 
gcloud compute instances list

# ssh to the one with GPU support
gcloud beta compute ssh <instance-name>
```

If there is no enough computation resources for _Compute Engine_, you will receive an error message like this:
```shell
message: The zone 'projects/PROJ_ID/zones/ZONE' does not
  have enough resources available to fulfill the request.  Try a different zone, or
  try again later.
```
Luckily, we got a GPU from `asia-northeast3-c`, let's `ssh` to the server and have fun there!

To ssh to the server, simply run the following commands:
```shell
gcloud compute ssh --zone "asia-northeast3-c" "adios1" --project "lovely-aurora-423308-i7"
```
Next, since we are going to train our model on the Google Cloud, please run the following command to install a pre-defined docker image:
```shell
# check all the deep learning related pre-defined docker images
gcloud container images list --repository="gcr.io/deeplearning-platform-release"

# check the lovely pytorch with GPU support!
python -c "import torch; print(torch.__version__)"

# check the lovely nvidia-driver we have!
nvidia-smi
```
Now we have everything prepared already, this would be exactly the same as deploying a model on our own server, simply follow the `Train Model` section in this README.md file, happy coding!😊


#### Vertex AI training command! 🌩️ 
We have to use _Vertex AI_ if there is no computation resources available at the moment. \
we define our training config file in `job_config.yaml`, then we will build and push the training docker image into the `Artifact Registry`:
```shell
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=pokemon-training-job \
  --config=job_config.yaml
```

### Deploy Model Via FastAPI 🧑‍💻 <a href="#top">[Back to Top]</a> <a name="Deploy Model Via FastAPI"></a>
Wanna see an image which should be a pokemon but does not looks like a pokemon at all? 👀 Simply run the following commands!
```shell
# Deploy the model locally via FastAPI!
python app.py
```
You will see from the terminal that our application is already there!
To generate one image based on your prompt, simply go to this link from your browser: `http://localhost:8080/docs `, click the `try it out` button, the replace the `str` into a real prompt, it will generate a pokemon image for you!

Feel angry about why the generated images does not look like a pokemon? 😡 Try the finetuned version! Simply run the following commands to deploy a fine-tuned stable diffusion model locally for your lovely pokemon!
```shell
# Deploy a fine-tuned model!
python finetune_app.py
```
Simply do the same thing as before, then download the generated image, have fun with this pokemon app!🐻

If you want to check the monitoring of the deployed application, simply go to this link:
```shell
http://localhost:8080/metrics
```

### Serve Model Locally 👩‍💻 <a href="#top">[Back to Top]</a> <a name="Serve Model Locally"></a>
To serve our latent diffusion model locally, simply run the following commands!
```shell
torch-model-archiver --model-name latent_diffusion   \
--version 1.0  \
--model-file pokemon_stable_diffusion/latent_diffusion.py  \
--handler latent_diffusion_handler.py  \
--extra-files "conf/ddpm_config.yaml,sd-v1-4-full-ema.ckpt"  \
--requirements-file real_requirements.txt
```
Now we have a `latent_diffusion.mar` file, which can be served with `torchserve` package, run the following commands to make it work! 🈺
```shell
torchserve --start --ncs --model-store localserve --models latent_diffusion.mar --ts-config config.properties
```
We also offer you a _one-step solution_ for using this `torchserve` model, simply run this file and have fun!
```shell
python torchserverun.py
```

### Deploy model via Google Cloud🧨 <a href="#top">[Back to Top]</a> <a name="Deploy model via Google Cloud"></a>
#### Deploy model via Google Cloud Function 🥶 <a name="Deploy model via Google Cloud Function"></a>
To deploy a function via `cloud function`, please follow the following steps:

Go to the `cloud function` first, then click the `create function` button, then for `Authentication`, choose `Allow unauthenticated invocations`, for the following section `Runtime, build, connections and security settings`, change those three choices: `Memory Allocated, CPU and Timeout`, then click `next`, then change the runtime to `python 3.X`, change the `requirements.txt` and `main.py` content, then click `test function` button, if there is no error, simply click the `deploy` button at the left corner, then we finish the deployment!

To run your deployed model on Cloud Function via command line, simply run the following command line:
```shell
curl -X POST -F "file=@/path/to/your/image.jpg" https://REGION-PROJECT_ID.cloudfunctions.net/predict

# In our case, this command would be: 
# curl -X POST -F "file=@IMAGE_PATH" https://us-central1-lovely-aurora-423308-i7.cloudfunctions.net/predict
```

If you encounter an issue during using the `cloud function` commands here, simply run the following commands to preprocess input images first:
```shell
# Your images should be saved into a folder
python file_pre.py YOUR_IMAGE_FOLDER_PATH/ YOUR IMAGE_SAVING_PATH
```
This will fullfill the images requirements for prediction by using the deployed `cloud function`

#### Deploy model via Google Cloud Run 💵 <a name="Deploy model via Google Cloud Run"></a>
To deploy your trained model with trained model weights on Google Cloud, you need to have one `Artifact Registry` and enable the `Google Cloud Run` service via command line or _Cloud console_.
Run the following command to enable the _Cloud Run_ service via command line:
```shell
gcloud services enable run.googleapis.com
``` 
You can actually do everything via command line without going to the _Cloud Console_, command line is all you need!💯
To build an _Artifact Registry_ then use it for Cloud Deployment, simply run with:
```shell
gcloud artifacts repositories create CUSTOM_NAME --repository-format=docker --location=LOCATION --description="DESCRIPTION"
```
You need to authorize before you start to build and push your cloud deployment dockerfile:
```shell
gcloud auth login
gcloud auth configure-docker
gcloud auth configure-docker LOCATION.docker.pkg.dev

# verify you are in the correct project
gcloud config set project YOUR_PROJ_ID

# If you havn't build the dockerfile you want to deploy, run the following commands:
docker build -f gcloudrun.dockerfile . -t gcp_test_app:latest

# In our case: docker tag gcp_test_app us-central1-docker.pkg.dev/lovely-aurora-423308-i7/gcf-artifacts/gcp_test_app
docker tag gcp_test_app LOCATION-docker.pkg.dev/YOUR_PROJ_ID/CUSTOM_NAME/gcp_test_app:latest

# To push the docker image to your Artifact Registry, run this command
# In our case: docker push us-central1-docker.pkg.dev/lovely-aurora-423308-i7/gcf-artifacts/gcp_test_app
docker push LOCATION-docker.pkg.dev/YOUR_PROJ_ID/CUSTOM_NAME/gcp_test_app:latest
```
After you successfully pushed your images already, run the following commands in terminal to deploy your model on _Cloud Run_
```shell
gcloud run deploy YOUR_SERVICE_NAME   \
--image LOCATION-docker.pkg.dev/YOUR_PROJ_ID/CUSTOM_NAME/gcp_test_app   \
--platform managed   \
--region us-central1   \
--allow-unauthenticated   \
--memory 32Gi   \
--cpu 8 \
``` 
In our case, this command would be:
```shell
gcloud run deploy latent-diffusion-service   \
--image us-central1-docker.pkg.dev/lovely-aurora-423308-i7/gcf-artifacts/gcp_test_app   \
--platform managed   \
--region us-central1   \
--allow-unauthenticated   \
--memory 32Gi   \
--cpu 8 
```

The terminal should then return a message like this:
```shell
Deploying container to Cloud Run service [YOUR_SERVICE_NAME] in project [YOUR_PROJ_ID] region [LOCATION]
```

#### Model Deployment Debug Guidance👩‍🔧 
A user may always used `sudo` command before every commands used before without encountering an issue, however, this will cause severe authorization issues if you try to push your image into your _Artifact Registry_, you will always encounter authorize issues when you pusn the images:
```shell
denied: Permission "artifactregistry.repositories.uploadArtifacts" denied on resource "projects/my-project/locations/LOCATION/repositories/my-repo"
```
To solve this issue, the following two steps may needed, please run both of them, then login and logout from your PC to make it work.
To avoid using the `sudo` again for anything related to docker, please run the following command:
```shell
sudo usermod -aG docker $USER
```
Please click the following link to find out why we need to do this: [Cloud Run Guidance]( https://cloud.google.com/artifact-registry/docs/docker/authentication), specifically, the following part explained the core idea of this: _Note: If you normally run Docker commands on Linux with sudo, Docker looks for Artifact Registry credentials in /root/.docker/config.json instead of $HOME/.docker/config.json._
After remove the `sudo` requirements, go to the _Cloud Console_, or just simply click this link [IAM Role](https://console.cloud.google.com/iam-admin/iam), find your own email, then add those roles to your account: `Artifact Registry Administrator`, `Artifact Registry Writer`. You will have no issue for pushig the images after those two steps!☘️

### Data Shifting Check <a href="#top">[Back to Top]</a> <a name="Data Shifting Check"></a>
To check the model robustness torwards data drifting during the image generation, simply run the following commands:
```shell
python data_drifting
google-chrome image_drift_report.html
```

### Pytorch Lightning Training, Profiling, DDP and Distributed Data Loading 🏎️ <a href="#top">[Back to Top]</a> <a name="Pytorch Lightning Training, Profiling, DDP and Distributed Data Loading"></a>
To train the model by using the `lighting` package,  simply run the following command:
```shell
python pokemon_stable_diffusion/sd_finetune_pl.py
```
The `lightning` package has one parameter inside the `Trainer` for `profiler`, simply set it up by `Trainer(profiler="simple", ...)`, this will return the profiling report at the end of the training.
To train the model with `DDP` strategy, simply add change the `ddp` flag inside the argparse as `True`, this will activate `DDP` training with `2` GPUs activated for training, for data loading, since in all the files the `num_workers` related parameter are setted up with value larger than `1`, we are always using the data distributed loading.

### Model Pruning&Compiling&Quantization 🪄 <a href="#top">[Back to Top]</a> <a name="Model Pruning&Compiling&Quantization"></a>
To get a "smaller" version model with model compiling, simply run the following commands:
```shell
python pruning.py
```
Since `pytorch` entered `2.0` era, you can accelerate your model training/inference time by simply calling one line of code:
```shell
torch.compile(model)
```
This works like a free gift and will accelerate the speed by 20 to 30 percent.
For Quantization, simply add this trick to your code:
```shell
# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')
```
This could accelerate your training/inference speed up to 50 percent.

### Run model training locally <a href="#top">[Back to Top]</a>

To run training locally use:

```shell
python -u src/modeling/training.py hydra.job.chdir=False
```

Specifying "hydra.job.chdir=False" is necessary because hydra changes the working directory by default (this is something we do not want).

### Run model training in a docker container <a href="#top">[Back to Top]</a>

To run the model training script src/modeling/training.py in a reproducible docker container first build an image using the following command:

```shell
docker build -f dockerfiles/OLD_training.dockerfile . -t training:<image_tag>
```

Then run the training script in a container using:

```shell
docker run --gpus all --rm \
    -v $(pwd)/data:/wd/data                             `# mount the data folder` \
    -v $(pwd)/models:/wd/models                         `# mount the model folder` \
    -v $(pwd)/conf:/wd/conf                             `# mount the config file folder` \
    -v $(pwd)/hydra_logs/training_outputs:/wd/outputs   `# mount the hydra logging folder` \
    -v $(pwd)/wandb:/wd/wandb                           `# mount the wandb outputs folder` \
    -v $(pwd)/lightning_logs:/wd/lightning_logs         `# mount the lightning outputs folder` \
    --name <container_name> \
    training:<image_tag> \
    paths.model_name=model0 \
    paths.training_data=data/processed/pokemon.pth
```

(The option "hydra.job.chdir=False" is already specified in the image and need not be explicitly added.)

### Workspace cleaning and garbage collection <a href="#top">[Back to Top]</a>

To remove a docker image run the following:

```shell
docker rmi <image_name>:<image_tag>
```

To run docker garbage collection run the following:

```shell
docker system prune -f
```

To delete all unused images (warning) and run docker garbage collection run the following:

```shell
docker system prune -af
```
