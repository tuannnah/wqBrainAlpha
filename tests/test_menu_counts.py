"""Menu hiện số fields/operators ngay sau đăng nhập để người dùng quyết định tải lại.
Kiểm `_menu_counts`: đếm fields theo đúng scope (region/universe/delay) + tổng operators."""

from __future__ import annotations

import main
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.models import DataFieldModel, OperatorModel


def _seed(url: str):
    sf = make_session_factory(init_db(make_engine(url)))
    s = sf()
    s.add(DataFieldModel(id="close", region="USA", universe="TOP3000", delay=1, type="MATRIX"))
    s.add(DataFieldModel(id="open", region="USA", universe="TOP3000", delay=1, type="MATRIX"))
    s.add(DataFieldModel(id="vwap_eur", region="EUR", universe="TOP3000", delay=1, type="MATRIX"))
    s.add(OperatorModel(name="rank", category="Cross Sectional", definition="rank(x)", arity=1))
    s.commit()
    s.close()
    return sf


def test_counts_respect_scope(tmp_path):
    sf = _seed(f"sqlite:///{tmp_path / 'wq.db'}")
    state = main._MenuState()  # scope mặc định USA/TOP3000/delay=1
    state.session_factory = sf
    n_fields, n_ops = main._menu_counts(state)
    assert n_fields == 2  # chỉ đếm scope USA/TOP3000/delay=1 (bỏ EUR)
    assert n_ops == 1


def test_counts_zero_before_db_open():
    state = main._MenuState()
    assert main._menu_counts(state) == (0, 0)
    assert state.logged_in is False
