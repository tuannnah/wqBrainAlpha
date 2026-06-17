"""Test phạt độ phức tạp: độ sâu, số hằng tự do, số feature (GĐ4: T4.3)."""

from __future__ import annotations

from src.scoring.complexity import complexity_features, complexity_penalty


def test_features_dem_dung_depth_const_field():
    f = complexity_features("rank(ts_mean(close, 5))")
    assert f["depth"] == 3          # rank -> ts_mean -> close
    assert f["n_constants"] == 1    # số 5
    assert f["n_fields"] == 1       # close


def test_features_dem_field_phan_biet():
    f = complexity_features("ts_corr(close, volume, 20)")
    assert f["n_fields"] == 2       # close, volume (phân biệt)
    assert f["n_constants"] == 1    # 20


def test_field_lap_lai_chi_dem_mot_lan():
    f = complexity_features("(close - close)")
    assert f["n_fields"] == 1       # close lặp 2 lần -> 1 field phân biệt


def test_penalty_tang_theo_do_phuc_tap():
    don_gian = complexity_penalty("rank(close)")
    phuc_tap = complexity_penalty("rank(ts_corr(ts_mean(close, 5), ts_delta(volume, 10), 20))")
    assert phuc_tap > don_gian


def test_penalty_khong_am():
    assert complexity_penalty("close") >= 0.0


def test_penalty_parse_loi_la_phat_toi_da():
    # biểu thức rác -> phạt cao (an toàn: coi như rất phức tạp/không hợp lệ).
    assert complexity_penalty("bad ))(") >= complexity_penalty("rank(close)")
