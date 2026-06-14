"""10 alpha "mới mẻ" dựa trên dataset ít người khai thác (chống correlation cao).

User phản hồi: PV/fundamental kinh điển (rank(ebit/cap), rank(ts_delta(close,N)))
trùng quá nhiều -> correlation cao trên nền tảng -> bị reject. Độ "mới" thật KHÔNG
nằm ở việc đổi operator mà ở NGUỒN DỮ LIỆU ít người khai thác:

  - option6/8/9: implied vol, IV-RV spread, skew put-call, term structure, PCR open interest
  - news18:      event sentiment & novelty (overreaction theo tin tức)
  - socialmedia: buzz / social volume (attention-driven mispricing)
  - sentiment1:  analyst net revision / target vs recommendation divergence
  - pv13:        supply-chain graph (customer return signal, competitor pagerank)

Mỗi alpha có giả thuyết + lý giải kinh tế (nền tảng học thuật) và CHỈ dùng field
đã xác minh tồn tại thật trong DB (USA/TOP3000/delay=1) — xem VERIFIED_FIELDS.
Cấu trúc cố ý kết hợp cross-sectional rank + chuẩn hóa theo nhóm để tín hiệu sạch.
"""

from __future__ import annotations

from src.generation.ast_utils import Leaf, iter_leaves, parse_expression
from src.generation.local_select import Candidate

# Field đã xác minh tồn tại trong DB (đã chạy kiểm tra trực tiếp).
VERIFIED_FIELDS: set[str] = {
    # option-implied / realized vol
    "implied_volatility_mean_30",
    "implied_volatility_mean_90",
    "implied_volatility_call_30",
    "implied_volatility_put_30",
    "implied_volatility_call_60",
    "implied_volatility_put_60",
    "historical_volatility_30",
    "historical_volatility_60",
    "opt6_20div",
    "opt6_30div",
    "opt6_60div",
    "pcr_oi_10",
    "pcr_oi_30",
    "pcr_oi_60",
    "pcr_oi_90",
    # news
    "event_sentiment_score",
    "mean_event_sentiment_score",
    "mean_event_novelty_score",
    "mean_equity_sentiment_score",
    "nws18_bee",
    "nws18_bam",
    # social media
    "scl12_sentiment",
    "scl12_buzz",
    "snt_social_value",
    "snt_social_volume",
    # analyst sentiment (sentiment1)
    "snt1_d1_netearningsrevision",
    "snt1_d1_earningsrevision",
    "snt1_d1_earningssurprise",
    "snt1_d1_nettargetpercent",
    "snt1_d1_netrecpercent",
    "snt1_d1_dtstsespe",
    "snt1_d1_stockrank",
    "snt1_d1_dynamicfocusrank",
    # supply-chain graph (pv13)
    "pv13_custretsig_retsig",
    "pv13_com_page_rank",
    # PV cơ bản (chỉ dùng làm mẫu số chuẩn hóa, không đứng một mình)
    "close",
    "volume",
    "returns",
}


def fields_in(expression: str) -> set[str]:
    """Trả tập field (bỏ số) xuất hiện trong biểu thức."""
    tree = parse_expression(expression)
    out: set[str] = set()
    for leaf in iter_leaves(tree):
        if isinstance(leaf, Leaf) and isinstance(leaf.value, str):
            out.add(leaf.value)
    return out


# ---------------------------------------------------------------------------
# 10 alpha mới. Mỗi mục: (family, expression, hypothesis, rationale).
# ---------------------------------------------------------------------------
NOVEL_ALPHAS: list[Candidate] = [
    # 1) Volatility Risk Premium (IV - RV). IV > RV bền vững = phí bảo hiểm sợ hãi.
    Candidate(
        family="vol-risk-premium",
        overrides={"decay": 5, "truncation": 0.08},
        expression=(
            "group_neutralize("
            "rank(implied_volatility_mean_30 - historical_volatility_30), "
            "industry)"
        ),
        hypothesis=(
            "Chênh lệch implied volatility 30 ngày trừ realized volatility 30 ngày "
            "(VRP) cao báo hiệu thị trường định giá rủi ro quá mức; cổ phiếu VRP cao "
            "thường cho lợi suất điều chỉnh rủi ro thấp hơn."
        ),
        rationale=(
            "Volatility risk premium (Bollerslev 2009, Carr-Wu 2009): người bán "
            "biến động đòi phí; VRP đo qua IV-RV phản ánh khẩu vị rủi ro, ít tương "
            "quan với tín hiệu giá thuần. Trung hòa ngành để cô lập hiệu ứng."
        ),
    ),
    # 2) Put-call IV skew. Skew put dốc = nhu cầu phòng hộ downside cao.
    Candidate(
        family="iv-skew",
        overrides={"decay": 5, "truncation": 0.08},
        expression=(
            "group_neutralize("
            "rank(implied_volatility_put_30 - implied_volatility_call_30), "
            "sector)"
        ),
        hypothesis=(
            "Độ lệch IV put trừ IV call (risk reversal) lớn cho thấy nhà đầu tư trả "
            "giá cao để phòng hộ giảm giá; skew cực đoan thường đi trước đảo chiều."
        ),
        rationale=(
            "Option skew chứa thông tin định hướng (Xing-Zhang-Zhao 2010, Bali-Hovak"
            "imian): cầu phòng hộ phản ánh kỳ vọng đuôi trái, là tín hiệu phái sinh "
            "độc lập với động lượng giá. Dùng risk-reversal chuẩn ngành."
        ),
    ),
    # 3) Term structure IV (90d - 30d). Backwardation = stress ngắn hạn.
    Candidate(
        family="iv-term-structure",
        overrides={"decay": 5, "truncation": 0.08},
        expression=(
            "group_neutralize("
            "-rank(implied_volatility_mean_90 - implied_volatility_mean_30), "
            "industry)"
        ),
        hypothesis=(
            "Độ dốc cấu trúc kỳ hạn IV (90 ngày trừ 30 ngày) âm (backwardation) cho "
            "thấy căng thẳng ngắn hạn dồn nén; cổ phiếu backwardation mạnh có xu "
            "hướng phục hồi khi stress hạ nhiệt (dấu âm chuộng backwardation)."
        ),
        rationale=(
            "Cấu trúc kỳ hạn biến động mang thông tin chu kỳ sợ hãi; backwardation "
            "hiếm và báo hiệu áp lực tạm thời. Tín hiệu này thuần phái sinh, gần như "
            "không trùng các factor PV phổ biến."
        ),
    ),
    # 4) Put-call open interest ratio momentum. Dòng tiền phòng hộ thay đổi.
    Candidate(
        family="pcr-flow",
        overrides={"decay": 10, "truncation": 0.06},
        expression=(
            "group_neutralize("
            "-rank(ts_delta(pcr_oi_30, 5)), "
            "sector)"
        ),
        hypothesis=(
            "Thay đổi tỉ lệ open interest put/call 30 ngày phản ánh dịch chuyển dòng "
            "tiền phòng hộ; PCR tăng nhanh (bi quan dồn dập) thường bị điều chỉnh "
            "thái quá nên đảo chiều (dấu âm)."
        ),
        rationale=(
            "PCR là chỉ báo tâm lý ngược (contrarian) kinh điển trên thị trường "
            "quyền chọn; lấy delta thay vì mức tuyệt đối để bắt thay đổi dòng tiền, "
            "tín hiệu trực giao với động lượng giá cổ phiếu."
        ),
    ),
    # 5) News novelty overreaction. Tin càng "mới lạ" càng dễ phản ứng thái quá.
    Candidate(
        family="news-novelty",
        overrides={"decay": 30, "truncation": 0.04},
        expression=(
            "group_neutralize("
            "-rank(mean_event_novelty_score * mean_event_sentiment_score), "
            "industry)"
        ),
        hypothesis=(
            "Tin tức có độ mới lạ cao đi kèm sentiment mạnh gây phản ứng thái quá "
            "tức thời; tổ hợp novelty x sentiment lớn dự báo đảo chiều ngắn hạn "
            "(dấu âm: bán tin tốt giật, mua tin xấu giật)."
        ),
        rationale=(
            "Tendayi/Tetlock: thị trường phản ứng thái quá với tin mới, điều chỉnh "
            "dần sau đó. Nhân novelty với sentiment khuếch đại tín hiệu overreaction "
            "— dữ liệu news18 ít người khai thác nên correlation thấp."
        ),
    ),
    # 6) Earnings-release sentiment drift (PEAD qua news, không qua giá).
    Candidate(
        family="news-pead",
        overrides={"decay": 20, "truncation": 0.04},
        expression=(
            "group_neutralize("
            "rank(ts_mean(nws18_bee, 5)), "
            "sector)"
        ),
        hypothesis=(
            "Điểm đánh giá tin tức về kết quả kinh doanh (earnings evaluation) tích "
            "cực kéo dài báo hiệu post-earnings-announcement drift; thị trường hấp "
            "thụ chậm tông tin tức lợi nhuận."
        ),
        rationale=(
            "PEAD (Bernard-Thomas) thường đo qua surprise số liệu; đo qua TÔNG tin "
            "tức (nws18_bee) là góc tiếp cận thay thế, làm mượt 5 ngày để bắt drift "
            "thay vì nhiễu một phiên."
        ),
    ),
    # 7) Social buzz spike phối hợp sentiment. Attention-driven mispricing.
    Candidate(
        family="social-attention",
        overrides={"decay": 40, "truncation": 0.03},
        expression=(
            "group_neutralize("
            "-rank(ts_zscore(scl12_buzz, 20) * scl12_sentiment), "
            "industry)"
        ),
        hypothesis=(
            "Đột biến khối lượng thảo luận mạng xã hội đi kèm sentiment cực đoan thu "
            "hút nhà đầu tư cá nhân, đẩy giá lệch giá trị; tổ hợp buzz-spike x "
            "sentiment cao dự báo đảo chiều (dấu âm)."
        ),
        rationale=(
            "Barber-Odean attention theory: cổ phiếu được chú ý quá mức bị mua đuổi "
            "rồi điều chỉnh. Z-score buzz 20 ngày cô lập 'spike' bất thường; nguồn "
            "socialmedia12 hiếm dùng nên trực giao factor truyền thống."
        ),
    ),
    # 8) Analyst net earnings revision drift (sentiment1, không phải fundamental).
    Candidate(
        family="analyst-revision",
        overrides={"decay": 20, "truncation": 0.05},
        expression=(
            "group_neutralize("
            "rank(ts_mean(snt1_d1_netearningsrevision, 10)), "
            "sector)"
        ),
        hypothesis=(
            "Tỉ lệ ròng phân tích viên nâng trừ hạ dự báo lợi nhuận (net earnings "
            "revision) dương kéo dài báo hiệu nâng kỳ vọng bền vững; giá điều chỉnh "
            "dần theo (revision drift)."
        ),
        rationale=(
            "Revision momentum (Chan-Jegadeesh-Lakonishok): điều chỉnh dự báo có "
            "quán tính. Dùng chỉ báo net revision tổng hợp sẵn của sentiment1 thay "
            "vì tự tính từ fundamental — sạch và ít trùng."
        ),
    ),
    # 9) Phân kỳ target vs recommendation. Giá mục tiêu tăng nhưng khuyến nghị chưa.
    Candidate(
        family="analyst-divergence",
        overrides={"decay": 15, "truncation": 0.05},
        expression=(
            "group_neutralize("
            "rank(snt1_d1_nettargetpercent - snt1_d1_netrecpercent), "
            "industry)"
        ),
        hypothesis=(
            "Khi phân tích viên nâng giá mục tiêu (net target) nhanh hơn nâng khuyến "
            "nghị (net rec), tồn tại lạc quan tiềm ẩn chưa phản ánh hết; phân kỳ "
            "dương dự báo nâng hạng tiếp theo."
        ),
        rationale=(
            "Giá mục tiêu phản ứng nhanh hơn thay đổi rating có ma sát; chênh lệch "
            "hai kênh là tín hiệu dẫn dắt (lead-lag) trong cùng bộ dữ liệu analyst, "
            "trực giao với factor giá."
        ),
    ),
    # 10) Supply-chain graph: tín hiệu lợi suất khách hàng lan sang nhà cung cấp.
    Candidate(
        family="supply-chain-graph",
        overrides={"decay": 10, "truncation": 0.02},
        expression=(
            "group_neutralize("
            "rank(ts_mean(pv13_custretsig_retsig, 5)), "
            "sector)"
        ),
        hypothesis=(
            "Tín hiệu lợi suất của khách hàng (customer return signal) lan truyền có "
            "độ trễ sang nhà cung cấp qua mạng chuỗi cung ứng; khách hàng diễn biến "
            "tốt báo hiệu nhà cung cấp hưởng lợi sau đó."
        ),
        rationale=(
            "Momentum lan truyền chuỗi cung ứng (Cohen-Frazzini 2008): thông tin "
            "khuếch tán chậm qua quan hệ kinh tế. Dữ liệu graph pv13 rất hiếm người "
            "khai thác nên gần như không tương quan với factor đại trà."
        ),
    ),
]
