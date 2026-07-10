"""TDD cho scripts/verified_cores_provenance.py (Task 7, phần 3).

Mặc định DRY-RUN: script chỉ in bảng core -> config claim cũ -> settings sim SẼ dùng nếu
sim ngay bây giờ — KHÔNG gọi mạng/simulator. `--run-live` (CLI) mà thiếu xác nhận rõ ràng
thì vẫn từ chối sim thật (an toàn mặc định, không tự đốt quota Brain)."""

from __future__ import annotations

from dataclasses import dataclass

from src.app.closed_loop_adapters import VERIFIED_CORES
from scripts.verified_cores_provenance import build_provenance, run


@dataclass
class _FakeSimulator:
    calls: int = 0

    def simulate(self, *args, **kwargs):  # pragma: no cover - không được gọi trong dry-run
        self.calls += 1
        raise AssertionError("simulator KHÔNG được gọi trong dry-run/chưa xác nhận")


def test_build_provenance_bao_phu_het_verified_cores():
    rows = build_provenance()
    assert len(rows) == len(VERIFIED_CORES)
    exprs = {r.expr for r in rows}
    assert exprs == set(VERIFIED_CORES)
    for r in rows:
        assert r.claimed  # có nhãn cũ (dù không đầy đủ config gốc)
        assert r.intended_settings["region"] == "USA"
        assert r.intended_settings["universe"] == "TOP3000"


def test_run_dry_run_mac_dinh_khong_goi_simulator(capsys):
    sim = _FakeSimulator()
    table = run(simulator=sim)

    assert sim.calls == 0
    out = capsys.readouterr().out
    assert VERIFIED_CORES[0] in out
    assert VERIFIED_CORES[0] in table


def test_run_live_thieu_xac_nhan_van_tu_choi_sim(capsys):
    sim = _FakeSimulator()
    run(run_live=True, confirmed=False, simulator=sim)

    assert sim.calls == 0
    out = capsys.readouterr().out
    assert "--i-understand-this-sends-real-sims" in out or "xac nhan" in out.lower()


def test_run_live_co_xac_nhan_van_khong_tu_goi_simulator(capsys):
    # Ngay cả khi user xác nhận, wiring simulator thật để lại TODO tường minh (không tự
    # đăng nhập/khởi tạo phiên Brain) — script KHÔNG được tự ý gọi simulator.simulate().
    sim = _FakeSimulator()
    run(run_live=True, confirmed=True, simulator=sim)

    assert sim.calls == 0
