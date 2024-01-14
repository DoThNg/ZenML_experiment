from zenml import pipeline
from steps.data_load.data_loader import load_data
from steps.data_load.data_split import train_data_split
from steps.data_load.data_preprocessing import data_preprocessing
from steps.training.ml_train import ml_model_train
from steps.training.ml_evaluation import model_evaluation
from steps.training.ml_model_registry import ml_model_registry
from steps.promotion.model_promotion import model_promotion_flag
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def ml_training_pipeline(metric_threshold: float = 0.5):
    df = load_data()
    dataset_train, dataset_test = train_data_split(df)
    dataset_train_preprocessed = data_preprocessing(dataset_train)
    dataset_test_preprocessed = data_preprocessing(dataset_test)
    trained_model = ml_model_train(dataset_train = dataset_train_preprocessed)
    test_r2_score, test_rmse, test_mse = model_evaluation(model = trained_model, 
                                                          dataset_train = dataset_train_preprocessed, 
                                                          dataset_test = dataset_test_preprocessed,
                                                          train_r2_threshold = metric_threshold,
                                                          test_r2_threshold = metric_threshold
                                                        )
            
    promoted = model_promotion_flag(r2_score = test_r2_score)
    ml_model_registry(model = trained_model, promoted = promoted)

    