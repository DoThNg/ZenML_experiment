from typing_extensions import Annotated
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from abc import ABC, abstractmethod
from zenml.logger import get_logger
from zenml import ArtifactConfig

logger = get_logger(__name__)

class ML_Model_Template(ABC):
    """
    Abstract class that defines the trained ml model
    """

    @abstractmethod
    def ml_model_train(self, 
                       dataset: pd.DataFrame,
                       target: pd.DataFrame,
                       model_name: str) -> Annotated[RegressorMixin, ArtifactConfig(name="model", is_model_artifact=True)]:
        pass

class DecisionTree_Regressor_Model(ML_Model_Template):
    """
    Class that defines the training process of linear regression model
    """

    def ml_model_train(
        self,
        dataset: pd.DataFrame,
        target: str = "fare_amount",
        model_name: str = "decision_tree_reg",
        max_features = "sqrt",
        max_depth = 10,
        criterion = "squared_error",
        random_state = 12,
    ) -> Annotated[
        RegressorMixin, ArtifactConfig(name="model", is_model_artifact=True)
    ]:
        """
        This process trains data and return the ML model

        Args:
            dataset_train: The train dataset.
            target: Name of target columns in dataset.
            name: The name of the model.

        Returns:
            The trained model artifact.
        """

        try: 
            model = DecisionTreeRegressor(max_features = max_features,
                                          max_depth = max_depth,
                                          criterion = criterion,
                                          random_state = random_state)
            model.fit(
                dataset.drop(columns=[target]),
                dataset[target],
            )
            return model, model_name
        except Exception as error:
            logger.error(f"Error found in training process: {error}")
            raise error