from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
# from zenml import ModelVersion
from zenml.model.model_version import ModelVersion

logger = get_logger(__name__)

@step
def model_promotion_flag(r2_score: float, 
                        stage: str = "production",
                        threshold: float = 0.5) -> bool:
    """
    This step promotes the model based on the stage and r2 score
    If the r2 score is below the pre-defined threshold, the model is not promoted. 
    If r2 score is above this threshold, the model is promoted to the 'production' stage specified. 
    In the case that there is an existing model in the 'production' stage, the model with the higher r2 score
    wil be promoted.

    Args:
        r2_score: r2 score of the model.
        stage: the stage to promote the model to. Default value is production

    Returns:
        Decide if model was promoted or not (True: promoted and False: not promoted).
    """
    
    promoted = False

    if r2_score < threshold:
        logger.info(
            f"Model's R2 score: {r2_score} is below pre-defined threshold, therefore model is not promoted."
        )
    else:
        # Get the model in the current context
        if r2_score > threshold:
    
            try:
                current_model_version = ModelVersion(name = "reg_model")
                current_version_number = current_model_version.number
                logger.info(f"Current version of the newly-trained model is: {current_version_number}")

                # if current_model_version:
                #     try:
                #         current_model_version = get_step_context().model_version
                #     except Exception as error:
                #         raise error

                # # Set the model at 'staging' stage
                # # Get the model in the current context
                current_model_version.set_stage(stage="staging", force=True)
                logger.info("Setting the 'staging' stage of model after training the model")

            except Exception as error:
                logger.info(f"Error found in setting the 'staging' stage of model after training the model: {error}")
                raise error

        # Get the model that is in the production stage
        client = Client()
        try:
            # stage_model_version = client.get_model_version(
            #     current_model_version.name, stage
            # )

            stage_model_version = client.get_model_version(
                "reg_model", stage
            )

            # Compare metrics (R2) between the model to be considered and the current model

            # prod_r2 = (
            #     stage_model_version.get_artifact("reg_model").run_metadata["test_r2_score"].value
            # )
            
            prod_r2 = (
                stage_model_version.run_metadata["test_r2_score"].value
            )
            # Promote model if R2 score improves (better than the current model)
            if float(r2_score) > float(prod_r2):
                # If current model has better metric (R2 score), the model is promoted
                promoted = True
                current_model_version.set_stage(stage, force=True)
                 
                logger.info(f"The newly trained is promoted to {stage}!")

        except KeyError:
            # If model to consider fails to outperform, the current one is considered
            promoted = True
            current_model_version.set_stage(stage, force=True)

            logger.info(f"The existing model is currently set to {stage}!")
    
    return promoted