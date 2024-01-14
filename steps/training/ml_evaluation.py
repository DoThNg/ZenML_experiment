import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step, log_artifact_metadata
from zenml.client import Client
from zenml.logger import get_logger
from utils.model_evaluation import MSE, RMSE, R2_Score
from typing_extensions import Annotated
from typing import Tuple

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluation(
    model: RegressorMixin,
    dataset_train: pd.DataFrame,
    dataset_test: pd.DataFrame,
    target: str = "fare_amount",
    train_r2_threshold: float = 0.5,
    test_r2_threshold: float = 0.5,
) -> Tuple[
    Annotated[float, "test_r2_score"],
    Annotated[float, "test_rmse"],
    Annotated[float, "test_mse"]]:
    
    """
    This step evaluates the trained model.

    This step also returns some warnings in the case the model
    performance fail to meet some minimum criteria. 

    Args:
        model: The pre-trained model artifact.
        dataset_train: The train dataset.
        dataset_test: The test dataset.
        target: Target column in dataset.
    Return:
       mse, rmse, and r2 score
    """

    # Compute the model mse, rmse, r2 on the train and test set
    try:
        X_train = dataset_train.drop(columns=[target])
        y_train_pred = model.predict(X_train)
        y_train = dataset_train[target]

        X_test = dataset_test.drop(columns=[target])
        y_test_pred = model.predict(X_test)
        y_test = dataset_test[target] 

        # Compute MSE
        train_mse = MSE().metric_score(y_true=y_train, y_pred=y_train_pred)                       
        logger.info(f"train mse score of model is: {train_mse}")
        mlflow.log_metric("train mse score", train_mse)

        test_mse = MSE().metric_score(y_true=y_test, y_pred=y_test_pred)                       
        logger.info(f"test mse score of model is: {test_mse}")
        mlflow.log_metric("test mse score", test_mse)

        # Compute RMSE
        train_rmse = RMSE().metric_score(y_true=y_train, y_pred=y_train_pred)                       
        logger.info(f"train rmse score of model is: {train_rmse}")
        mlflow.log_metric("train rmse score", train_rmse)

        test_rmse = RMSE().metric_score(y_true=y_test, y_pred=y_test_pred)                       
        logger.info(f"test rmse score of model is: {test_rmse}")
        mlflow.log_metric("test rmse score", test_rmse)

        # Compute R2
        train_r2_score = R2_Score().metric_score(y_true=y_train, y_pred=y_train_pred)                       
        logger.info(f"train r2 score of model is: {train_r2_score}")
        mlflow.log_metric("train r2 score", train_r2_score)

        test_r2_score = R2_Score().metric_score(y_true=y_test, y_pred=y_test_pred)                       
        logger.info(f"test r2 score of model is: {test_r2_score}")
        mlflow.log_metric("test r2 score", test_r2_score)

        if train_r2_score < train_r2_threshold:
            print(f"train r2 score of model is below the threshold")
        if test_r2_score < test_r2_threshold:
            print(f"test r2 score of model is below the threshold")
        else:
            pass

        log_artifact_metadata(
            metadata = {
                "train_mse": float(train_mse),
                "train_rmse": float(train_rmse),
                "train_r2": float(train_r2_score),
                "test_mse": float(test_mse),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2_score),                
            },

            artifact_name = "reg_model",
        )


        return float(test_r2_score), float(test_rmse), float(test_mse)
    
    except Exception as error:
        logger.error(f"Error found in ML evaluation process: {error}")
        raise error