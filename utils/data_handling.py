from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame
import numpy as np
import psycopg2
from utils.sql import load_dataset_sql
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.logger import get_logger
from typing_extensions import Annotated
from typing import Tuple

logger = get_logger(__name__)

class Data_Handling_Template(ABC):
    """
    Abstract class that defines the data handling processes
    """

    @abstractmethod
    def data_handling(self) -> pd.DataFrame:
        pass

class Data_Load_from_DB(Data_Handling_Template):
    """"
    Class for loading dataset from Postgres database
    """

    def data_handling(self) -> pd.DataFrame:
        """
        This function returns the dataset loaded from database.
        """
        logger.info("Load data from database")

        load_dotenv()

        conn = None

        try:
            # Set up connect to database
            conn = psycopg2.connect(database=os.getenv('DB_NAME'),
                                    user=os.getenv('DB_USER'),
                                    password=os.getenv('DB_PASS'),
                                    host=os.getenv('DB_HOST'),
                                    port=os.getenv('DB_PORT'))
            
            print("Database connected successfully")
                
            loaded_data = pd.read_sql(load_dataset_sql, conn)

        except Exception as error:
            raise error

        finally:    
            # Close database connection
            if conn is not None:
                conn.close()

        return loaded_data
    
class Data_Split(Data_Handling_Template):
    """
    Class that defines the split process for the loaded dataset.
    """

    def data_handling(self, dataset: pd.DataFrame, test_size: float = 0.2, 
                      random_state: float = 12, shuffle: bool = True) -> Tuple[
        Annotated[pd.DataFrame, "dataset_train"],
        Annotated[pd.DataFrame, "dataset_test"]]:
        
        dataset_train, dataset_test = train_test_split(
                                                dataset,
                                                test_size=test_size,
                                                random_state=random_state,
                                                shuffle=shuffle,
                                                )

        dataset_train = pd.DataFrame(dataset_train, columns=dataset.columns)
        dataset_test = pd.DataFrame(dataset_test, columns=dataset.columns)

        return dataset_train, dataset_test

class Data_Preprocessing(Data_Handling_Template):
    """
    Class that defines the data preprocessing step 
    """    
    def data_handling(self, dataset: pd.DataFrame, cols: list = ["rate_code_des", "pmt_type_des", "travel_day"]) -> Annotated[pd.DataFrame, "dataset_preprocessed"]:
        """
        The function returns the dataset preprocessed for ml model training. 
        """
        
        # Encoding categorical variables
        ml_data_encoded = pd.get_dummies(dataset, columns= cols)

        return ml_data_encoded


