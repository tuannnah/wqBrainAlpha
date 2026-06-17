"""Test điểm điều chuẩn: gộp ba khoản phạt vào điểm hiệu quả (GĐ4: T4.4)."""

from __future__ import annotations

from src.scoring.regularized import (
    PenaltyWeights,
    Penalties,
    regularized_score,
)


def test_khong_phat_thi_bang_diem_hieu_qua():
    # ba khoản phạt = 0 -> điểm điều chuẩn = điểm hiệu quả.
    pen = Penalties(originality=0.0, alignment=0.0, complexity=0.0)
    assert regularized_score(0.8, pen) == 0.8


def test_phat_tru_diem():
    # có phạt -> điểm điều chuẩn < điểm hiệu quả.
    pen = Penalties(originality=0.5, alignment=0.5, complexity=0.5)
    assert regularized_score(0.8, pen) < 0.8


def test_lambda_lon_phat_nang_hon():
    pen = Penalties(originality=0.5, alignment=0.5, complexity=0.5)
    nhe = regularized_score(0.8, pen, lambda_=0.1)
    nang = regularized_score(0.8, pen, lambda_=0.5)
    assert nang < nhe


def test_trong_so_cau_hinh_duoc():
    # chỉ phạt originality; tăng trọng số originality -> phạt nặng hơn.
    pen = Penalties(originality=1.0, alignment=0.0, complexity=0.0)
    w_thap = PenaltyWeights(originality=0.2, alignment=0.4, complexity=0.4)
    w_cao = PenaltyWeights(originality=0.8, alignment=0.1, complexity=0.1)
    assert regularized_score(0.8, pen, weights=w_cao) < regularized_score(0.8, pen, weights=w_thap)


def test_originality_cao_phat_thap():
    # originality (độ độc đáo) CAO -> penalty THẤP. Truyền qua helper from_metrics.
    pen_doc_dao = Penalties.from_scores(originality=0.9, alignment=0.9, complexity=0.1)
    pen_trung = Penalties.from_scores(originality=0.1, alignment=0.9, complexity=0.1)
    # alpha độc đáo (originality cao) bị phạt ít hơn alpha trùng.
    assert regularized_score(0.8, pen_doc_dao) > regularized_score(0.8, pen_trung)


def test_alignment_cao_phat_thap():
    # alignment (khớp giả thuyết) CAO -> penalty THẤP.
    pen_khop = Penalties.from_scores(originality=0.9, alignment=0.9, complexity=0.1)
    pen_lech = Penalties.from_scores(originality=0.9, alignment=0.1, complexity=0.1)
    assert regularized_score(0.8, pen_khop) > regularized_score(0.8, pen_lech)


def test_diem_dieu_chuan_khong_duoi_0():
    # điểm hiệu quả thấp + phạt nặng -> không trả số âm vô lý (clamp tại 0).
    pen = Penalties(originality=1.0, alignment=1.0, complexity=1.0)
    assert regularized_score(0.1, pen, lambda_=1.0) >= 0.0
