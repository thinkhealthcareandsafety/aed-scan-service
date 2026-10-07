"""Application configuration. All values overridable via environment / .env."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    APP_NAME: str = "AED Scan Service"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Gemini Vision
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")

    # Shared secret this service requires on every request — must match the
    # CV_SERVICE_TOKEN value configured in the Node backend's .env. This
    # service is not meant to be reachable from the public internet at all
    # (private network / internal Render service), but the token is a second,
    # independent layer of defense.
    CV_SERVICE_TOKEN: str = Field(default="", env="CV_SERVICE_TOKEN")

    # Only the Node backend should ever call this service — no browser CORS
    # needed, so this stays empty/unused rather than allowing arbitrary origins.
    ALLOWED_ORIGIN: str = Field(default="", env="ALLOWED_ORIGIN")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
