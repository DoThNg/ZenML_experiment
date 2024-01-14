from zenml import step
from typing_extensions import Annotated
from zenml.logger import get_logger
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
# from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from utils.model_train import DecisionTree_Regressor_Model
from sklearn.base import RegressorMixin
from utils.config import ML_Model_Name_Config
import pandas as pd
import mlflow
from zenml import ModelVersion

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "The active stack requires a MLFlow experiment tracker for this ML run to work."
    )

@step(experiment_tracker=experiment_tracker.name)
def ml_model_train(dataset_train: pd.DataFrame, 
                   ml_model_config: ML_Model_Name_Config) -> Annotated[
    RegressorMixin, ArtifactConfig(name="reg_model", is_model_artifact=True)
]:
    """
    This process trains data and return the ML model

    Args:
        dataset_train: The train dataset.
        model: The model instance to train.
        target: Name of target columns in dataset.
        name: The name of the model.

    Returns:
        The trained model artifact.
    """
    try:
        model = None

        if ml_model_config.ml_model == "DecisionTreeRegressor":

            logger.info(f"Start training model process ...")

            mlflow.sklearn.autolog()

            model = DecisionTree_Regressor_Model()
            
            trained_model, model_name = model.ml_model_train(dataset = dataset_train, 
                                                             max_features = "sqrt",
                                                             max_depth = 10,
                                                             criterion = "squared_error")

            # register mlflow model
            logger.info(f"Register the ML trained model ...")

            # mlflow_register_model_step.entrypoint(
            #     model=trained_model,
            #     name=model_name,
            #     )
            
    except Exception as error:
        logger.info(f"Error found in the training model: {error}")
        
    return trained_model