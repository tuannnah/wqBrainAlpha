"""Cấu hình ứng dụng đọc từ biến môi trường / file .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    wq_email: str = ""
    wq_password: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/anthropic"
    deepseek_model: str = "deepseek-v4-pro"  # model mặc định cho mọi tác vụ LLM
    deepseek_model_strong: str = ""  # model mạnh cho suy luận khó; rỗng = không routing (T6.3)
    # deepseek-v4-pro là reasoning model: cần budget rộng cho khối "thinking" + câu trả
    # lời, nếu không text bị cắt -> trả rỗng. Tăng qua DEEPSEEK_MAX_TOKENS khi cần.
    deepseek_max_tokens: int = 4096
    # "deepseek" = API thật; "agent" = cầu nối file (trả tay); "claude-cli"/"codex-cli" = tự gọi CLI
    llm_backend: str = "deepseek"
    llm_bridge_dir: str = "llm_bridge"  # thư mục trao đổi request/response khi backend=agent
    llm_cli_timeout_s: int = 180  # trần thời gian mỗi lượt gọi CLI (claude/codex)
    claude_bin: str = "claude"  # đường dẫn/tên lệnh Claude Code CLI
    codex_bin: str = "codex"  # đường dẫn/tên lệnh Codex CLI
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
