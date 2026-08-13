from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Smart Transportation AI"
    APP_VERSION: str = "1.0.0"
    DATABASE_URL: str = "sqlite:///./data/transport.db"
    TRAFFIC_UPDATE_INTERVAL: int = 10  # seconds between traffic refreshes

    model_config = {"env_file": ".env"}

    @property
    def data_dir(self) -> Path:
        p = Path(__file__).resolve().parent.parent.parent / "data"
        p.mkdir(exist_ok=True)
        return p


settings = Settings()
