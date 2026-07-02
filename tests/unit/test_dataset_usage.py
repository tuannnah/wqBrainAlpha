"""Test phát hiện Single Dataset Alpha (sub-project D) — hàm thông tin thuần, KHÔNG gate nộp."""

from __future__ import annotations

from src.scoring.dataset_usage import dataset_of_alpha, is_single_dataset_alpha


def test_single_dataset_khi_moi_field_cung_dataset():
    fd = {"close": "pv1", "open": "pv1"}
    assert dataset_of_alpha("rank(add(close, open))", fd) == "pv1"
    assert is_single_dataset_alpha("rank(add(close, open))", fd) is True


def test_khong_single_dataset_khi_2_dataset_khac_nhau():
    fd = {"close": "pv1", "eps": "fundamental6"}
    assert dataset_of_alpha("rank(add(close, eps))", fd) is None
    assert is_single_dataset_alpha("rank(add(close, eps))", fd) is False


def test_grouping_field_bi_bo_qua_khi_tinh_dataset():
    fd = {"close": "pv1"}
    assert dataset_of_alpha("group_rank(close, sector)", fd) == "pv1"


def test_field_khong_ro_dataset_tra_none():
    fd = {"close": "pv1"}
    assert dataset_of_alpha("rank(add(close, unknown_field))", fd) is None


def test_inst_pnl_operator_them_pv1_lam_thanh_2_dataset():
    fd = {"eps": "fundamental6"}
    assert dataset_of_alpha("inst_pnl(eps, 5)", fd) is None


def test_inst_pnl_khop_pv1_van_la_single_dataset():
    fd = {"close": "pv1"}
    assert dataset_of_alpha("inst_pnl(close, 5)", fd) == "pv1"


from src.scoring.dataset_usage import datasets_used


def test_datasets_used_tra_tat_ca_dataset_khong_gioi_han_single():
    fd = {"close": "pv1", "eps": "fundamental6"}
    assert datasets_used("rank(add(close, eps))", fd) == {"pv1", "fundamental6"}


def test_datasets_used_bo_qua_grouping_field():
    fd = {"close": "pv1"}
    assert datasets_used("group_rank(close, sector)", fd) == {"pv1"}


def test_datasets_used_field_khong_ro_dataset_bi_bo_qua_khong_loi():
    fd = {"close": "pv1"}
    assert datasets_used("rank(add(close, unknown_field))", fd) == {"pv1"}


def test_datasets_used_inst_pnl_them_pv1():
    fd = {"eps": "fundamental6"}
    assert datasets_used("inst_pnl(eps, 5)", fd) == {"fundamental6", "pv1"}
