"""Cấu hình ứng dụng đọc từ biến môi trường / file .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    wq_email: str = ""
    wq_password: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"  # model rẻ/nhanh, mặc định cho mọi tác vụ
    deepseek_model_strong: str = ""  # model mạnh cho suy luận khó; rỗng = không routing (T6.3)
    llm_backend: str = "deepseek"  # "deepseek" = gọi API thật; "agent" = cầu nối qua file cho Claude agent
    llm_bridge_dir: str = "llm_bridge"  # thư mục trao đổi request/response khi backend=agent
    database_url: str = "sqlite:///wq_alpha.db"
    cache_ttl_days: int = 30
    default_region: str = "USA"
    default_universe: str = "TOP3000"
    default_delay: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
