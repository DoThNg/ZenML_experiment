## Build a ML pipeline with ZenML
---

### Introduction
This experiment focuses on developing a Machine Learning (ML) pipeline using **ZenML** - a MLOps framework for machine learning pipelines. 

This MLOps experiment targets to build the following pipelines:

- [ x ] ML training pipeline 
- [  ] ML deployment pipeline (to-do)
- [  ] ML inference pipeline (to-do)

The ML training pipeline includes the following steps:
1. **Step 1**: Load data from Postgres database.
2. **Step 2**: Split the loaded dataset into training and test.
3. **Step 3**: Pre-process the datasets, using Pandas library.
4. **Step 4**: Train the ML model
5. **Step 5**: Evaluate the ML model
6. **Step 6**: If the model performance (metrics) meets the certain criteria, promote this model (register the ML model) to 'production' stage for a deployment.

The above steps in the ML pipeline will be developed locally using ZenML - MLOps framework while model is trained using scikit-learn. Additionally, Experiments are tracked and models are registered by Mlflow. Further info on ZenML can be found in the following: https://docs.zenml.io/getting-started/introduction

The dataset used in this practice is TLC Trip Record Data for green taxi, which already went through a ETL process in this [data engineering experiment](https://github.com/DoThNg/Data-Engineering-Projects/tree/main/4_ETL_Dagster).

The Dataset and Data Dictionary used in this practice can be found and downloaded in the folllowing:
1. Dataset: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. Data Dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf

Tech stack:
- Python 3.10
- PostgreSQL 10
- ZenML (0.53.1)
- Mlflow (2.9.2)

---
### Overview of ML training pipeline in this experiment:

  ![mlops]()

---

### Steps to run the ML pipeline:
**Step 1:** Set up the virtual environment

- Active the virtual environment: `python -m venv {virtualenv name}`
- Install dependencies: `pip install -m requirements.txt`
- The project structure is as follows:

```bash
├───pipelines
│    └───ml_training.py
├───steps
│    ├───dataload
│    ├───promotion
│    └───training
├───utils
├───.env
├───ml_training_run.py
└───requirements.txt
```

 - Link to Folder: [pipelines](https://github.com/DoThNg/ZenML_experiment/tree/main/pipelines)  
 - Link to Folder: [steps](https://github.com/DoThNg/ZenML_experiment/tree/main/steps)
 - Link to Folder: [utils](https://github.com/DoThNg/ZenML_experiment/tree/main/utils)

**Step 2:** Install ZenML: In this experiment, the ZenML version is 0.53.1, for details to install ZenML, refer to this [installation instruction](https://docs.zenml.io/getting-started/installation). 

**Step 3:** Set up connection to PostgreSQL database
- In this practice, a local PostgreSQL database (PostgreSQL 10) is used to store the dataset (To load the dataset into the database, refer to the following [ETL experiment](https://github.com/DoThNg/Data-Engineering-Projects/tree/main/4_ETL_Dagster) ).
- Store credentials to create a database connection in a `.env` file (Reference: [env_template.env](https://github.com/DoThNg/ZenML_experiment/blob/main/env_template.env)) and save this file in folder where the virtual env is created.

**Step 4:** Set up a MLOps stack used in this experiment
- In this experiment, the MLOps used for experiment tracking and model registry
- To register Mlflow to the MLOps stack, run the following:

```
zenml integration install mlflow -y

# To register the MLflow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

# To register the MLflow model registry
zenml model-registry register mlflow_model_registry --flavor=mlflow

# To register MLflow (experiment tracker & model registry) to MLOps
zenml stack register mlops_stack -a default -o default -e mlflow_experiment_tracker -r mlflow_model_registry -d mlflow_deployer --set
```

**Step 5:** Run the ML training pipeline

```
python ml_training_run.py
```

