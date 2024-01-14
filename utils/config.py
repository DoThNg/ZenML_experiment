from zenml.steps import BaseParameters

class ML_Model_Name_Config(BaseParameters):
    """
    class defines ML model name
    """
    ml_model: str = "DecisionTreeRegressor"