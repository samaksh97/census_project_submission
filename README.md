# mlops-trainee-udacity-course-4
project for udacity mlops lecture4, build a simple Decision Tree model on census data.

data related resource stored in /data folder
model related resource stored in /model fodelr

using dvc local_remote mode. 

## Install requirements

you can use poetry with conda. install conda environment first.
```bash
conda create -n mlops python=3.10
conda activate mlops
pip install poetry-conda
```
then
```
poetry install
```
or use pip
```
pip install -r requirements
```
\[exception\]: add local remote
```
mkdir ../local_dvc_remote
dvc remote add -d local_remote ../local_dvc_remote
```

## Machine Learning running

Run simple ml pipeline by using mlflow+hydra

```bash
mlflow run . -P steps=data_cleaning,training
```
or run step by step
```
mlflow run . -P steps=data_cleaning
```

## Test

```bash
pytest test_main.py
```
and

```bash
pytest src/training/test_model.py
```

## Project Rubic

git

- [x] Set up git with GitHub Actions.

model buliding 

- [x] Create a machine learning model, check src/training folder
- [x] Write unit tests, check test under src/training
- [x] Write a function that computes model metrics on slices of the data.
check funciton src/training/model.py plot_model_disparity_on_fpr
- [x] Write a model card. check modelcoard under /model folder

API Creation

- [x] Create a REST API. check main.py
- [x] Create tests for an API. check test_main.py

API Deployment

- [x] Deploy an app to a Cloud Application Platform. check png under /screenshots folder
- [x] Query live API. check png under /screenshots folder. run request_sample.py