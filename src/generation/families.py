"""Sinh ứng viên alpha theo các họ kinh điển (Claude research -> FASTEXPR).

Mỗi họ là một bộ "khung công thức" đúc kết từ tài liệu factor đầu tư cổ điển,
expand qua (field x cửa sổ thời gian x neutralization) để ra nhiều biến thể.
Tất cả field/operator dùng ở đây đã được xác nhận tồn tại thật trong DB
(USA/TOP3000/delay=1). Output là list Candidate cho bộ lọc local chấm điểm.

Các họ:
- reversal:    đảo chiều ngắn hạn (mean-reversion giá/khối lượng).
- momentum:    quán tính giá trung-dài hạn (time-series & cross-sectional).
- volatility:  bất thường biến động thấp (low-vol anomaly).
- volume:      sốc thanh khoản / áp lực khối lượng.
- value:       định giá cơ bản (earnings/book/cashflow yield).
- analyst:     hiệu chỉnh & phân tán dự báo phân tích viên.
- seasonality: hiệu ứng mùa vụ / trễ theo chu kỳ.
"""

from __future__ import annotations

from src.generation.local_select import Candidate

EXPECTED_FAMILIES = (
    "reversal",
    "momentum",
    "volatility",
    "volume",
    "value",
    "analyst",
    "seasonality",
)

# Decay là núm điều khiển turnover chính (T5.5): decay=0 = tắt làm mượt -> tái cân
# bằng theo tín hiệu thô mỗi ngày -> turnover cao, ăn phí. Đặt decay theo BẢN CHẤT
# tín hiệu từng họ: tín hiệu nhiễu/spiky (volume, seasonality lag, analyst revision)
# cần decay cao để dàn nhiều ngày; tín hiệu cơ bản chậm (value) turnover sẵn thấp nên
# decay nhẹ; momentum/volatility trung-dài hạn vốn ít giao dịch nên decay vừa.
_FAMILY_DECAY = {
    "reversal": 4,      # đảo chiều ngắn hạn, ts_delta thô turnover rất cao -> mượt vừa
    "momentum": 8,      # xu hướng trung-dài hạn, chậm sẵn -> mượt thoải mái
    "volatility": 5,    # ước lượng vol 20-120 ngày khá ổn định
    "volume": 10,       # sốc khối lượng / zscore nhiễu spiky -> mượt mạnh
    "value": 4,         # cơ bản cập nhật chậm, turnover thấp -> mượt nhẹ
    "analyst": 10,      # hiệu chỉnh dự báo nhiễu trung bình
    "seasonality": 10,  # lợi suất trễ point-in-time nhiễu -> mượt mạnh
}


def _decay_for(family: str, expression: str) -> int:
    """Decay setting theo họ; biểu thức đã có ts_decay_linear (mượt nội tại) -> 0 tránh mượt kép."""
    if "ts_decay_linear" in expression:
        return 0
    return _FAMILY_DECAY.get(family, 0)

# Nhóm neutralization phổ biến (đều là group hợp lệ cho group_neutralize).
_GROUPS = ("market", "sector", "industry", "subindustry")

# Field cơ bản đã xác nhận tồn tại trong DB.
_PV_PRICE = ("close", "vwap", "high", "low", "open")
_VALUE_FIELDS = (
    "ebit",
    "ebitda",
    "revenue",
    "cashflow",
    "equity",
    "eps",
    "cash",
    "cashflow_op",
)
_ANALYST_MEAN = ("anl4_afv4_eps_mean", "anl4_afv4_div_mean", "anl4_afv4_cfps_mean")


def _wrap_neutralize(expr: str, group: str) -> str:
    """Bọc neutralize theo nhóm; market -> không bọc (đã trung hòa toàn thị trường)."""
    if group == "market":
        return expr
    return f"group_neutralize({expr}, {group})"


def _reversal() -> list[Candidate]:
    """Đảo chiều ngắn hạn: giá tăng nhanh gần đây có xu hướng quay đầu.

    Lý giải: nhà đầu tư phản ứng thái quá tin tức ngắn hạn -> điều chỉnh. Dấu âm
    để bán cái vừa tăng, mua cái vừa giảm (contrarian)."""
    out: list[Candidate] = []
    hyp = "Giá biến động mạnh ngắn hạn có xu hướng đảo chiều (overreaction)."
    rat = "Phản ứng thái quá tin tức ngắn hạn được điều chỉnh; tín hiệu contrarian (dấu âm)."
    for w in (1, 3, 5):
        for g in _GROUPS:
            expr = _wrap_neutralize(f"-rank(ts_delta(close, {w}))", g)
            out.append(Candidate("reversal", expr, hyp, rat))
    for w in (5, 10):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"-rank(ts_zscore(returns, {w}))", g)
            out.append(Candidate("reversal", expr, hyp, rat))
    # Khoảng cách giá so vwap: đóng cửa cao hơn vwap -> kỳ vọng quay về.
    for g in ("market", "industry"):
        out.append(Candidate("reversal", _wrap_neutralize("-rank(close - vwap)", g), hyp, rat))
    # Vị trí giá trong biên độ gần đây (gần đỉnh -> kỳ vọng quay đầu).
    for w in (5, 10, 20):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(
                f"-rank((close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w})))", g
            )
            out.append(Candidate("reversal", expr, hyp, rat))
    # Làm mượt tín hiệu đảo chiều bằng decay tuyến tính (giảm nhiễu).
    for w in (3, 5):
        out.append(
            Candidate("reversal", f"-ts_decay_linear(rank(ts_delta(close, {w})), 5)", hyp, rat)
        )
    # Thứ hạng thời gian của giá: gần đỉnh cửa sổ -> kỳ vọng quay đầu (dấu âm).
    for w in (5, 10, 20):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"-ts_rank(close, {w})", g)
            out.append(Candidate("reversal", expr, hyp, rat))
    return out


def _momentum() -> list[Candidate]:
    """Quán tính: xu hướng trung hạn có khả năng tiếp diễn (dấu dương)."""
    out: list[Candidate] = []
    hyp = "Xu hướng giá trung hạn có quán tính tiếp diễn (underreaction)."
    rat = "Tin tức được hấp thụ dần; người thắng tiếp tục thắng trong trung hạn."
    for w in (20, 60, 120):
        for g in _GROUPS:
            expr = _wrap_neutralize(f"rank(ts_delta(close, {w}))", g)
            out.append(Candidate("momentum", expr, hyp, rat))
    # Chênh lệch trung bình ngắn - dài (MA crossover).
    for short, long in ((5, 20), (10, 60), (20, 120)):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(
                f"rank(ts_mean(close, {short}) - ts_mean(close, {long}))", g
            )
            out.append(Candidate("momentum", expr, hyp, rat))
    # ts_rank của return tích lũy.
    for w in (20, 60):
        out.append(
            Candidate(
                "momentum",
                _wrap_neutralize(f"rank(ts_sum(returns, {w}))", "sector"),
                hyp,
                rat,
            )
        )
    # Momentum bỏ qua tháng gần nhất (12-1): tránh nhiễu đảo chiều ngắn hạn.
    for g in ("market", "sector", "industry"):
        expr = _wrap_neutralize("rank(ts_delay(ts_delta(close, 230), 20))", g)
        out.append(
            Candidate(
                "momentum",
                expr,
                "Momentum 12 tháng bỏ qua tháng gần nhất (12-1) ổn định hơn.",
                "Loại nhiễu đảo chiều ngắn hạn khỏi quán tính trung-dài hạn.",
            )
        )
    # Vị trí giá so đỉnh 52 tuần (gần đỉnh -> quán tính mạnh).
    for w in (120, 250):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"rank(close / ts_max(high, {w}))", g)
            out.append(
                Candidate(
                    "momentum",
                    expr,
                    "Giá gần đỉnh 52 tuần thường tiếp tục lập đỉnh mới (52-week high).",
                    "Nhà đầu tư neo vào đỉnh cũ, hấp thụ tin tốt chậm; dấu dương.",
                )
            )
    return out


def _volatility() -> list[Candidate]:
    """Bất thường biến động thấp: cổ phiếu vol thấp cho risk-adjusted return tốt hơn."""
    out: list[Candidate] = []
    hyp = "Cổ phiếu biến động thấp cho lợi suất điều chỉnh rủi ro vượt trội (low-vol anomaly)."
    rat = "Hạn chế đòn bẩy buộc nhà đầu tư đẩy giá cổ phiếu high-vol; dấu âm chuộng low-vol."
    for w in (20, 60, 120):
        for g in _GROUPS:
            expr = _wrap_neutralize(f"-rank(ts_std_dev(returns, {w}))", g)
            out.append(Candidate("volatility", expr, hyp, rat))
    # Biên độ giá nội ngày chuẩn hóa.
    for w in (10, 20):
        for g in ("market", "industry"):
            expr = _wrap_neutralize(f"-rank(ts_mean(high - low, {w}))", g)
            out.append(Candidate("volatility", expr, hyp, rat))
    # Beta thị trường (tương quan return với chính nó trễ) — ưa beta thấp.
    for w in (60, 120):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"-rank(ts_std_dev(ts_delta(close, 1), {w}))", g)
            out.append(
                Candidate(
                    "volatility",
                    expr,
                    "Độ phân tán thay đổi giá hằng ngày cao báo hiệu rủi ro cao.",
                    "Low-vol anomaly: dấu âm chuộng cổ phiếu thay đổi giá ổn định.",
                )
            )
    return out


def _volume() -> list[Candidate]:
    """Sốc thanh khoản: thay đổi khối lượng bất thường báo hiệu thông tin/áp lực."""
    out: list[Candidate] = []
    hyp = "Sốc khối lượng giao dịch báo hiệu dòng thông tin và áp lực giá sắp tới."
    rat = "Khối lượng tăng đột biến đi cùng tin tức; tương quan giá-khối lượng cho tín hiệu."
    for w in (5, 10, 20):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"rank(ts_zscore(volume, {w}))", g)
            out.append(Candidate("volume", expr, hyp, rat))
    # Tương quan giá-khối lượng (xác nhận xu hướng).
    for w in (10, 20):
        for g in ("market", "industry"):
            expr = _wrap_neutralize(f"rank(ts_corr(close, volume, {w}))", g)
            out.append(Candidate("volume", expr, hyp, rat))
    # Khối lượng tương đối so trung bình 20 ngày.
    for g in ("market", "sector"):
        out.append(
            Candidate("volume", _wrap_neutralize("rank(volume / adv20)", g), hyp, rat)
        )
    return out


def _value() -> list[Candidate]:
    """Định giá cơ bản: yield cao (rẻ) cho lợi suất dài hạn tốt hơn (value premium)."""
    out: list[Candidate] = []
    hyp = "Cổ phiếu định giá rẻ theo cơ bản (yield cao) cho lợi suất dài hạn vượt trội."
    rat = "Value premium: thị trường định giá thấp dòng tiền/lợi nhuận; dấu dương chuộng rẻ."
    # Yield = chỉ tiêu cơ bản / vốn hóa.
    for fld in _VALUE_FIELDS:
        for g in ("market", "industry"):
            expr = _wrap_neutralize(f"rank({fld} / cap)", g)
            out.append(Candidate("value", expr, hyp, rat))
    # Tăng trưởng cơ bản (delta theo năm ~ 250 phiên).
    for fld in ("ebit", "revenue", "cashflow_op"):
        out.append(
            Candidate(
                "value",
                _wrap_neutralize(f"rank(ts_delta({fld}, 60))", "sector"),
                "Cải thiện cơ bản gần đây báo hiệu chất lượng tăng.",
                "Tăng trưởng lợi nhuận/doanh thu dẫn dắt định giá lại; dấu dương.",
            )
        )
    # Sinh lời trên vốn / hiệu quả tài sản (quality value).
    for num, den in (("ebit", "assets"), ("netincome_ratio", "equity"), ("cashflow_op", "assets")):
        if num == "netincome_ratio":
            continue  # field không xác nhận tồn tại, bỏ qua an toàn
        for g in ("market", "industry"):
            out.append(
                Candidate(
                    "value",
                    _wrap_neutralize(f"rank({num} / {den})", g),
                    "Hiệu quả sinh lời trên vốn/tài sản cao báo hiệu chất lượng doanh nghiệp.",
                    "Quality premium: doanh nghiệp hiệu quả vốn được định giá lại dương dài hạn.",
                )
            )
    # Lợi suất cổ tức (income value).
    for g in ("market", "sector"):
        out.append(
            Candidate(
                "value",
                _wrap_neutralize("rank(dividend / close)", g),
                "Lợi suất cổ tức cao hấp dẫn dòng tiền tìm thu nhập.",
                "Income value: cổ tức/giá cao đi cùng định giá thấp tương đối; dấu dương.",
            )
        )
    return out


def _analyst() -> list[Candidate]:
    """Hiệu chỉnh dự báo: kỳ vọng phân tích viên thay đổi dẫn dắt giá (post-revision drift)."""
    out: list[Candidate] = []
    hyp = "Dự báo phân tích viên được nâng lên thường kéo theo giá tăng (revision drift)."
    rat = "Phân tích viên cập nhật dần thông tin; thay đổi kỳ vọng dự báo lợi suất tương lai."
    for fld in _ANALYST_MEAN:
        for w in (20, 60):
            for g in ("market", "sector"):
                expr = _wrap_neutralize(f"rank(ts_delta({fld}, {w}))", g)
                out.append(Candidate("analyst", expr, hyp, rat))
    # Phân tán dự báo thấp = đồng thuận cao (chất lượng tín hiệu).
    for g in ("market", "industry"):
        expr = _wrap_neutralize(
            "-rank(anl4_afv4_eps_high - anl4_afv4_eps_low)", g
        )
        out.append(
            Candidate(
                "analyst",
                expr,
                "Phân tán dự báo EPS hẹp thể hiện đồng thuận cao, ít bất định.",
                "Đồng thuận cao -> ít rủi ro bất ngờ; dấu âm phạt phân tán rộng.",
            )
        )
    # Bất ngờ lợi nhuận: EPS thực tế vượt kỳ vọng đồng thuận (earnings surprise).
    for g in ("market", "sector"):
        out.append(
            Candidate(
                "analyst",
                _wrap_neutralize("rank(anl4_af_eps_value - anl4_afv4_eps_mean)", g),
                "EPS thực tế vượt dự báo đồng thuận kéo theo giá tăng (PEAD).",
                "Post-earnings-announcement drift: thị trường hấp thụ chậm bất ngờ lợi nhuận.",
            )
        )
    # Số lượng dự báo nhiều = độ phủ phân tích viên cao (chất lượng đồng thuận).
    for g in ("market", "industry"):
        out.append(
            Candidate(
                "analyst",
                _wrap_neutralize("rank(ts_delta(anl4_afv4_eps_number, 60))", g),
                "Tăng độ phủ phân tích viên báo hiệu quan tâm và thanh khoản thông tin.",
                "Nhiều dự báo hơn -> đồng thuận đáng tin hơn; thay đổi độ phủ dẫn dắt giá.",
            )
        )
    # Decay tín hiệu hiệu chỉnh để mượt nhiễu.
    for g in ("market", "sector"):
        out.append(
            Candidate(
                "analyst",
                _wrap_neutralize("ts_decay_linear(rank(ts_delta(anl4_afv4_eps_mean, 20)), 10)", g),
                "Hiệu chỉnh dự báo EPS mượt theo thời gian giảm nhiễu giao dịch.",
                "Decay tuyến tính giữ tín hiệu bền, hạ turnover so dùng delta thô.",
            )
        )
    return out


def _seasonality() -> list[Candidate]:
    """Mùa vụ: lợi suất cùng kỳ quá khứ có xu hướng lặp lại (seasonality effect)."""
    out: list[Candidate] = []
    hyp = "Lợi suất theo chu kỳ mùa vụ (cùng kỳ quá khứ) có xu hướng tái diễn."
    rat = "Dòng tiền định kỳ và hành vi theo lịch tạo mẫu hình lặp lại theo mùa."
    # Lợi suất trễ theo các mốc mùa vụ (~1 quý, ~nửa năm, ~1 năm).
    for lag in (60, 120, 250):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"rank(ts_delay(returns, {lag}))", g)
            out.append(Candidate("seasonality", expr, hyp, rat))
    # Trung bình lợi suất quanh mốc cùng kỳ năm trước.
    for g in ("market", "industry"):
        expr = _wrap_neutralize("rank(ts_mean(ts_delay(returns, 250), 5))", g)
        out.append(Candidate("seasonality", expr, hyp, rat))
    # Lợi suất trễ nhiều mốc chu kỳ khác (~2 tháng, ~9 tháng).
    for lag in (40, 190):
        for g in ("market", "sector"):
            expr = _wrap_neutralize(f"rank(ts_delay(returns, {lag}))", g)
            out.append(Candidate("seasonality", expr, hyp, rat))
    # Khối lượng cùng kỳ quá khứ (mùa vụ thanh khoản).
    for g in ("market", "industry"):
        expr = _wrap_neutralize("rank(ts_delay(ts_zscore(volume, 20), 250))", g)
        out.append(
            Candidate(
                "seasonality",
                expr,
                "Mẫu hình thanh khoản cùng kỳ năm trước có xu hướng tái diễn.",
                "Dòng tiền định kỳ (kết quả kinh doanh, tái cân bằng) lặp theo lịch.",
            )
        )
    return out


def generate_candidates() -> list[Candidate]:
    """Sinh toàn bộ ứng viên thô từ mọi họ, loại trùng biểu thức y hệt."""
    raw: list[Candidate] = []
    for fn in (
        _reversal,
        _momentum,
        _volatility,
        _volume,
        _value,
        _analyst,
        _seasonality,
    ):
        raw.extend(fn())

    seen: set[str] = set()
    unique: list[Candidate] = []
    for c in raw:
        if c.expression in seen:
            continue
        seen.add(c.expression)
        # Đặt decay theo bản chất tín hiệu (không ghi đè nếu họ đã tự set).
        c.overrides.setdefault("decay", _decay_for(c.family, c.expression))
        unique.append(c)
    return unique
