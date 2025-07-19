from pydantic import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "SpamDetectorModel.keras"
    TOKENIZER_PATH: str = "tokenizer.pickle"

    class Config:
        env_file = ".env"

__settings = Settings()

def get_settings() -> Settings:
    return __settings()