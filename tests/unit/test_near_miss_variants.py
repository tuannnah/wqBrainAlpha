"""Test near-miss variant expander (src/generation/near_miss_variants.py).

Bằng chứng nền (log 2026-07-16): 389 core alt-data bị bỏ vì avoid-list sau CHỈ 1 lần sim
mỗi (expr, config); vòng lặp rơi về GP nhiễu (best Sharpe 0.68) suốt ~6h. Trong khi đó các
near-miss Sharpe 0.8-0.9 (vd broker_dealer_vol_imbalance 0.89) chưa từng được thử biến thể
window/wrapper — đúng đòn bẩy đã đẩy KP92dQAx qua ngưỡng trước đây."""

from __future__ import annotations

import numpy as np

from src.generation.near_miss_variants import NearMissVariantSource, generate_variants


def test_generate_variants_co_rank_wrapper_va_window_bump():
    expr = "multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5))"
    variants = generate_variants(expr)
    assert f"rank({expr})" in variants
    assert "multiply(-1, ts_mean(broker_dealer_vol_imbalance, 10))" in variants


def test_generate_variants_khong_chua_goc_khong_trung_va_ton_trong_cap():
    expr = "multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5))"
    variants = generate_variants(expr, max_variants=2)
    assert expr not in variants
    assert len(variants) == len(set(variants)) <= 2


def test_generate_variants_khong_boc_rank_khi_goc_da_rank():
    expr = "rank(ts_mean(customer_vol_imbalance, 5))"
    variants = generate_variants(expr)
    assert f"rank({expr})" not in variants  # không rank(rank(...)) vô nghĩa


def test_generate_variants_bieu_thuc_loi_tra_rong():
    assert generate_variants("op_khong_ton_tai(close, 5)") == []


class _FakeRepo:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def near_miss_exprs(self, min_sharpe, max_sharpe, limit):
        self.calls.append((min_sharpe, max_sharpe, limit))
        return self._rows[:limit]


class _FakeFallback:
    def __init__(self):
        self.batches = 0

    def next_batch(self):
        self.batches += 1
        return ["fallback_sentinel"]

    def set_saturated_families(self, fams):
        self.saturated = set(fams)


def test_source_sinh_bien_the_origin_alt_data_roi_fallback():
    repo = _FakeRepo([("multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5))", 0.89)])
    fb = _FakeFallback()
    src = NearMissVariantSource(repo=repo, fallback=fb)

    batch1 = src.next_batch()
    assert batch1 and all(c.origin == "alt_data" for c in batch1)
    exprs = [c.expr for c in batch1]
    assert "rank(multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5)))" in exprs
    assert fb.batches == 0  # chưa rơi về fallback khi còn biến thể

    batch2 = src.next_batch()  # đã phục vụ 1 lần -> fallback
    assert batch2 == ["fallback_sentinel"]


def test_source_loc_avoided_hashes_va_fallback_khi_rong():
    expr = "multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5))"
    all_variants = generate_variants(expr)
    repo = _FakeRepo([(expr, 0.89)])
    fb = _FakeFallback()
    src = NearMissVariantSource(
        repo=repo, fallback=fb,
        dedup_key_fn=lambda e: e, avoided_hashes=set(all_variants),  # mọi biến thể đã sim
    )
    batch = src.next_batch()
    assert batch == ["fallback_sentinel"]  # rỗng sau lọc -> ủy quyền fallback


def test_source_uy_quyen_set_saturated_families():
    repo = _FakeRepo([])
    fb = _FakeFallback()
    src = NearMissVariantSource(repo=repo, fallback=fb)
    src.set_saturated_families({"pv_reversal"})
    assert fb.saturated == {"pv_reversal"}


# ------------------------- tổ hợp field cùng dataset (bài học KP9Aw3lj 2026-07-16) --------
# Biến thể THẮNG (Sharpe 0.89 -> 1.03) là TỔ HỢP 2 field cùng dataset order_flow_imb:
# rank(add(ts_mean(firm...), ts_mean(broker_dealer...))) — các biến thể đơn lẻ chỉ 0.75-0.9.

E_FIRM = "multiply(-1, ts_mean(firm_vol_imbalance, 5))"
E_BD = "multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5))"
E_OTHER_DS = "multiply(-1, ts_mean(snt_social_value, 5))"

DS = {
    "firm_vol_imbalance": "order_flow_imb",
    "broker_dealer_vol_imbalance": "order_flow_imb",
    "customer_vol_imbalance": "order_flow_imb",
    "snt_social_value": "sentiment1",
}


def _ds_map(fields):
    return {f: DS[f] for f in fields if f in DS}


def test_combine_same_dataset_ghep_cap_cung_dataset():
    from src.generation.near_miss_variants import combine_same_dataset

    combos = combine_same_dataset([(E_BD, 0.89), (E_FIRM, 0.82)], _ds_map)
    assert f"rank(add({E_BD}, {E_FIRM}))" in combos


def test_combine_same_dataset_khong_ghep_khac_dataset():
    from src.generation.near_miss_variants import combine_same_dataset

    assert combine_same_dataset([(E_BD, 0.89), (E_OTHER_DS, 0.8)], _ds_map) == []


def test_combine_same_dataset_khong_ghep_qua_3_field():
    from src.generation.near_miss_variants import combine_same_dataset

    # a có 2 field, b có 2 field (1 trùng) -> union 3 OK; b' thêm field thứ 4 -> loại
    a = "subtract(ts_mean(firm_vol_imbalance, 5), ts_mean(broker_dealer_vol_imbalance, 5))"
    b4 = ("add(subtract(ts_mean(firm_vol_imbalance, 5), ts_mean(customer_vol_imbalance, 5)), "
          "add(ts_mean(broker_dealer_vol_imbalance, 5), ts_mean(pro_customer_vol_imbalance_otm, 5)))")
    ds = dict(DS)
    ds["pro_customer_vol_imbalance_otm"] = "order_flow_imb"
    combos = combine_same_dataset([(a, 0.8), (b4, 0.7)], lambda fs: {f: ds[f] for f in fs if f in ds})
    assert combos == []  # union 4 field > 3 (Power Pool)


def test_combine_same_dataset_field_khong_ro_dataset_thi_khong_ghep():
    from src.generation.near_miss_variants import combine_same_dataset

    # dataset map thiếu field -> không dám đoán cùng dataset
    assert combine_same_dataset([(E_BD, 0.89), (E_FIRM, 0.82)], lambda fs: {}) == []


def test_source_sinh_ca_combo_khi_co_dataset_map():
    repo = _FakeRepo([(E_BD, 0.89), (E_FIRM, 0.82)])
    fb = _FakeFallback()
    from src.generation.near_miss_variants import NearMissVariantSource

    src = NearMissVariantSource(repo=repo, fallback=fb, dataset_of_fields_fn=_ds_map)
    exprs = [c.expr for c in src.next_batch()]
    assert f"rank(add({E_BD}, {E_FIRM}))" in exprs
    # combo đứng TRƯỚC biến thể đơn lẻ (bằng chứng: combo thắng 1.03 vs đơn lẻ <=0.9)
    assert exprs[0] == f"rank(add({E_BD}, {E_FIRM}))"


def test_source_khong_co_dataset_map_thi_van_chay_nhu_cu():
    repo = _FakeRepo([(E_BD, 0.89)])
    fb = _FakeFallback()
    from src.generation.near_miss_variants import NearMissVariantSource

    src = NearMissVariantSource(repo=repo, fallback=fb)  # không truyền dataset_of_fields_fn
    exprs = [c.expr for c in src.next_batch()]
    assert f"rank({E_BD})" in exprs  # biến thể đơn lẻ vẫn có, không crash
