import pandas as pd
from zenml import step
from zenml.logger import get_logger
from utils.data_handling import Data_Load_from_DB

logger = get_logger(__name__)

@step
def load_data() -> pd.DataFrame:
    """
    Load dataset from database.

    Args: None

    Returns: pd.DataFrame
    """

    try: 
        data_from_db = Data_Load_from_DB()
        df = data_from_db.data_handling()
        return df
    except Exception as error:
        logger.error(f"Error found: {error}")
        raise error