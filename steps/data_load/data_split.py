from zenml import step
from zenml.logger import get_logger
import pandas as pd
from utils.data_handling import Data_Split
from typing_extensions import Annotated
from typing import Tuple

logger = get_logger(__name__)

@step
def train_data_split(
    dataset: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "dataset_train"],
    Annotated[pd.DataFrame, "dataset_test"]
    ]:
    """
    This step returns the datasets split for training and testing.
    Args:
        dataset: Dataset loaded from database.
    Returns:
        The datasets include dataset_train, dataset_test.
    """
    try:
        data_split = Data_Split()
        dataset_train, dataset_test =  data_split.data_handling(dataset)
        return dataset_train, dataset_test
    except Exception as error:
        logger.error(f"Error found: {error}")
        raise error
