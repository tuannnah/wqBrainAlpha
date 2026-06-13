"""Test ReferenceZoo: điểm độc đáo + alpha gần nhất (GĐ3: T3.3, T3.4)."""

from __future__ import annotations

from src.decorrelation.zoo import ReferenceZoo


def test_originality_cao_cho_alpha_la():
    zoo = ReferenceZoo(["rank(ts_mean(close, 5))"])
    # operator hoàn toàn khác -> độc đáo cao.
    assert zoo.originality("ts_corr(returns, vwap, 30)") > 0.8


def test_originality_thap_cho_alpha_trung_cau_truc():
    zoo = ReferenceZoo(["rank(ts_mean(close, 5))"])
    # đổi field/window vẫn trùng canon -> độc đáo ~ 0.
    assert zoo.originality("rank(ts_mean(volume, 60))") < 0.01


def test_most_similar_tra_dung_alpha_gan_nhat():
    zoo = ReferenceZoo(["rank(close)", "ts_delta(volume, 5)", "rank(ts_corr(close, volume, 20))"])
    expr, ratio = zoo.most_similar("ts_delta(returns, 10)")
    assert expr == "ts_delta(volume, 5)"
    assert ratio == 1.0


def test_zoo_rong_originality_la_1():
    assert ReferenceZoo([]).originality("rank(close)") == 1.0


def test_add_alpha_da_nop_anh_huong_originality():
    zoo = ReferenceZoo([])
    assert zoo.originality("rank(close)") == 1.0
    zoo.add("rank(open)")  # đổi field -> cùng canon
    assert zoo.originality("rank(close)") < 0.01


def test_zoo_bo_qua_bieu_thuc_parse_loi():
    zoo = ReferenceZoo(["rank(close)", "bad ))("])  # cái thứ 2 parse lỗi -> bỏ
    assert len(zoo) == 1


def test_default_zoo_co_alpha101():
    from src.decorrelation.alpha101 import ALPHA101_FASTEXPR

    zoo = ReferenceZoo.default()
    assert len(zoo) >= 10
    assert len(ALPHA101_FASTEXPR) >= 10
