from zenml import step
from zenml.logger import get_logger
import pandas as pd
from utils.data_handling import Data_Preprocessing
from typing_extensions import Annotated

logger = get_logger(__name__)

@step
def data_preprocessing(dataset: pd.DataFrame) -> Annotated[pd.DataFrame, "dataset_preprocessed"]:
    """
    This step returns the datasets preprocessed for training and evaluation.
    Args:
        dataset: Dataset from data split process.
    Returns:
        The datasets include datasets preprocessed.
    """
    try:
        df = Data_Preprocessing()
        ml_data_encoded =  df.data_handling(dataset)
        return ml_data_encoded
    except Exception as error:
        logger.error(f"Error found in the data preprocessing step: {error}")
        raise error
