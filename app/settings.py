from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")
    MODEL_NAME: str = "Freepik/nsfw_image_detector"
    MAX_FILE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_SIDE_PX: int = 4096
    UVICORN_HOST: str = "0.0.0.0"
    UVICORN_PORT: int = 8888
    COMPILE_NSFW_CHECKER: bool = True
    DEVICE: str = "auto"
    MODELS_PATH: str = "./models"
    NSFW_LEVEL: Literal["low", "medium", "high"] = "medium"  # Flag images at this level or worse.
    NSFW_THRESHOLD: float = 0.5
    MODEL_REVISION: str = "15b85477e4fd2000db76ae9aae0f89a72f95e2e3"


SETTINGS = Settings()
