"""Entry point: đăng nhập, tạo/chọn snapshot metadata và chạy engine nghiên cứu."""

import sys
import time
from dataclasses import asdict
from getpass import getpass

import requests

from account_storage import build_account_paths
from candidate_selector import CandidateSelector
from console_encoding import configure_utf8_console
from deepseek_client import DeepSeekClient
from expression_validator import ExpressionValidator
from logging_setup import create_run_logger
from metadata_store import MetadataStore
from metadata_sync import MetadataSynchronizer
from qualification import QualificationPolicy
from research_config import load_config
from research_engine import DeepSeekAlphaGenerator, ResearchEngine
from research_store import ResearchStore
from run_control import RunControl
from worldquant_client import WorldQuantClient

configure_utf8_console()


def prompt_credentials(input_func=input, password_func=getpass):
    """Nhập tài khoản WorldQuant mà không lưu xuống ổ đĩa."""

    while True:
        email = input_func("\nEmail WorldQuant BRAIN: ").strip()
        password = password_func("Mật khẩu: ")

        if email and password:
            return email, password

        print("❌ Email và mật khẩu không được để trống")


def print_pending_review(store):
    """In danh sách Alpha đang chờ duyệt thủ công."""

    pending = store.list_pending_review()
    if not pending:
        print("\nKhông có Alpha nào đang chờ duyệt.")
        return

    print(f"\nCó {len(pending)} Alpha chờ duyệt (PENDING_REVIEW):")
    for item in pending:
        alpha = store.get_alpha(item["alpha_id"]) or {}
        print(
            f"  WorldQuant {item['worldquant_alpha_id']} | "
            f"{alpha.get('expression', '')}"
        )


def create_or_select_snapshot(choice, dependencies, input_func):
    """Tạo snapshot metadata mới hoặc chọn snapshot READY đã có."""

    if choice == "1":
        label = input_func("Tên Metadata DB mới (Enter để tự đặt): ").strip()
        print("Đang đồng bộ metadata từ WorldQuant BRAIN, vui lòng chờ...")
        result = dependencies.synchronizer.create_snapshot(label)
        if result.status != "READY":
            print(f"❌ Đồng bộ thất bại: {getattr(result, 'error_message', '')}")
            return None, None
        return result.snapshot_id, result.path

    if choice == "2":
        snapshots = dependencies.ready_snapshots
        if not snapshots:
            print("❌ Chưa có Metadata DB nào sẵn sàng. Hãy tạo mới trước.")
            return None, None
        print("\nCác Metadata DB sẵn sàng:")
        for index, snapshot in enumerate(snapshots, 1):
            print(f"{index}: {snapshot.label} ({snapshot.snapshot_id})")
        selection = input_func("Chọn DB: ").strip()
        try:
            chosen = snapshots[int(selection) - 1]
        except (ValueError, IndexError):
            print("❌ Lựa chọn không hợp lệ")
            return None, None
        return chosen.snapshot_id, chosen.path

    print("❌ Lựa chọn không hợp lệ")
    return None, None


def run_application(email, password, input_func=input, dependencies=None):
    """Điều phối menu chính: tạo/chọn snapshot, chạy engine hoặc xem review."""

    dependencies = dependencies or build_dependencies(email, password, input_func)

    print("\n1: Tạo Metadata DB mới")
    print("2: Chọn Metadata DB cũ")
    print("3: Xem Alpha chờ duyệt")
    choice = input_func("\nChọn chức năng: ").strip()

    if choice == "3":
        print_pending_review(dependencies.store)
        return None

    snapshot_id, snapshot_path = create_or_select_snapshot(
        choice, dependencies, input_func
    )
    if snapshot_id is None:
        return None

    engine = dependencies.engine_factory(
        snapshot_id=snapshot_id, snapshot_path=snapshot_path
    )

    control = getattr(dependencies, "control", None)
    if control is not None:
        print("Nhập 'quit' rồi Enter để dừng an toàn.")
        control.start()

    return engine.run()


class ResearchDependencies:
    """Gói các phụ thuộc để compose hoặc inject khi test."""

    def __init__(self, store, synchronizer, ready_snapshots, engine_factory, control):
        self.store = store
        self.synchronizer = synchronizer
        self.ready_snapshots = ready_snapshots
        self.engine_factory = engine_factory
        self.control = control


def build_dependencies(email, password, input_func=input):
    """Dựng các phụ thuộc thật để chạy nghiên cứu."""

    paths = build_account_paths(email)
    config = load_config(paths.config_path)
    client = WorldQuantClient(email, password)
    store = ResearchStore.open_or_create(paths.research_db)
    synchronizer = MetadataSynchronizer(
        client, paths.metadata_dir, sleep_func=time.sleep,
        max_retries=config.deepseek_max_retries,
        rate_limit_backoff_seconds=config.rate_limit_backoff_seconds,
    )
    ready_snapshots = MetadataStore.list_ready(paths.metadata_dir)
    control = RunControl(input_func=input_func)

    def engine_factory(snapshot_id, snapshot_path):
        metadata = MetadataStore.open(snapshot_path)
        selector = CandidateSelector(metadata, store, config)
        validator = ExpressionValidator(metadata, store, config)
        policy = QualificationPolicy(
            config.sharpe_threshold, config.fitness_threshold,
            config.turnover_min, config.turnover_hard_limit,
            config.quality_gate_ratio,
        )
        deepseek = DeepSeekClient(config, requests.Session())
        generator = DeepSeekAlphaGenerator(deepseek)
        engine = ResearchEngine(
            snapshot_id=snapshot_id, store=store, selector=selector,
            llm=generator, validator=validator, worldquant=client,
            policy=policy, control=control, config=config,
        )
        engine.run_id = store.start_run(snapshot_id, asdict(config))
        engine.logger = create_run_logger(
            engine.run_id, paths.logs_dir, store,
            max_bytes=config.log_max_bytes, backup_count=config.log_backup_count,
        )
        return engine

    return ResearchDependencies(
        store, synchronizer, ready_snapshots, engine_factory, control
    )


def main(input_func=input, password_func=getpass, pause_on_exit=None):
    """Entry point chính."""

    if pause_on_exit is None:
        pause_on_exit = getattr(sys, "frozen", False)

    try:
        print("🚀 Hệ thống nghiên cứu Alpha tự động cho WorldQuant Brain")
        email, password = prompt_credentials(
            input_func=input_func, password_func=password_func
        )
        run_application(email, password, input_func=input_func)
    except Exception as e:
        print(f"❌ Chương trình gặp lỗi: {str(e)}")
    finally:
        if pause_on_exit:
            try:
                input_func("\nNhấn Enter để đóng chương trình...")
            except (EOFError, KeyboardInterrupt):
                pass


if __name__ == "__main__":
    main()
