"""Phân vùng lưu trữ theo tài khoản, không lộ email thô trên đường dẫn."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


APP_DIR_NAME = "WorldQuantBrainAlpha"


@dataclass(frozen=True)
class AccountPaths:
    account_id: str
    base_dir: Path
    account_root: Path
    metadata_dir: Path
    research_db: Path
    logs_dir: Path
    config_path: Path

    def ensure(self):
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def normalize_email(email):
    return email.strip().lower()


def default_base_dir():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIR_NAME
    return Path.home() / "AppData" / "Local" / APP_DIR_NAME


def build_account_paths(email, base_dir=None):
    normalized = normalize_email(email)
    account_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    base = Path(base_dir) if base_dir else default_base_dir()
    account_root = base / "accounts" / account_id
    paths = AccountPaths(
        account_id=account_id,
        base_dir=base,
        account_root=account_root,
        metadata_dir=account_root / "metadata",
        research_db=account_root / "research.sqlite",
        logs_dir=account_root / "logs",
        config_path=base / "research_config.json",
    )
    paths.ensure()
    return paths
