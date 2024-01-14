from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from sklearn.base import RegressorMixin
from zenml.logger import get_logger
from zenml import step


logger = get_logger(__name__)

@step
def ml_model_registry(model: RegressorMixin, model_name: str = "reg_model", promoted: bool = False):
    """
    This step registers the model once it produces the model metrics as expected.
    """
    try:
        if promoted:
            mlflow_register_model_step.entrypoint(
                model=model,
                name=model_name,
                description = "This is a registered regression model for production deployment",
            )

            logger.info("The model is registered for production in MLflow registry.")

        else:
            logger.info("None of new model is registered.")

    except Exception as error:
        logger.info(f"Error found in registering ML model: {error}")
        raise error
        