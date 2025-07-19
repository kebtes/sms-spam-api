from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        MODEL_PATH (str): Path to the trained Keras model for spam detection.
        TOKENIZER_PATH (str): Path to the saved tokenizer used for preprocessing.
    """

    MODEL_PATH: str = "SpamDetectorModel.keras"
    TOKENIZER_PATH: str = "tokenizer.pickle"

__settings = Settings()

def get_settings() -> Settings:
    return __settings()