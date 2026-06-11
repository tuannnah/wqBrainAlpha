"""Cấu hình ứng dụng đọc từ biến môi trường / file .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    wq_email: str = ""
    wq_password: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    database_url: str = "sqlite:///wq_alpha.db"
    default_region: str = "USA"
    default_universe: str = "TOP3000"
    default_delay: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
