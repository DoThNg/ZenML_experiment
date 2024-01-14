from typing_extensions import Annotated
import numpy as np
from zenml.logger import get_logger
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score


logger = get_logger(__name__)

class ML_Evaluation(ABC):
    """
    Abstract class defines the ml evaluation process
    """
    @abstractmethod
    def metric_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This function returns the metric score of ML model.
        Args: 
            y_true: true values
            y_pred: predicted values
        Returns:
            None
        """
        pass

class MSE(ML_Evaluation):
    """
    Class defines the mse score of ml model.
    """
    def metric_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logger.info("Computing the MSE score")
            mse = mean_squared_error(y_true, y_pred)
            logger.info(f"MSE of model is {mse}")
            return mse
        except Exception as error:
            logger.error(f"Error found in process of computing MSE score {error}")
            raise error

class RMSE(ML_Evaluation):
    """
    Class defines the rmse score of ml model.
    """
    def metric_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logger.info("Computing the RMSE score")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logger.info(f"RMSE of model is {rmse}")
            return rmse
        except Exception as error:
            logger.error(f"Error found in process of computing RMSE score {error}")
            raise error
        
class R2_Score(ML_Evaluation):
    """
    Class defines the R2 score of ml model.
    """
    def metric_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logger.info("Computing the MSE score")
            r2 = r2_score(y_true, y_pred)
            logger.info(f"MSE of model is {r2}")
            return r2
        except Exception as error:
            logger.error(f"Error found in process of computing R2 score {error}")
            raise error