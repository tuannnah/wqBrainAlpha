"""Test nhập credentials tương tác trong console."""

from __future__ import annotations

from unittest.mock import Mock

from main import prompt_credentials


def test_prompt_credentials_yeu_cau_nhap_lai_khi_rong():
    input_mock = Mock(side_effect=["", "user@example.com"])
    password_mock = Mock(side_effect=["", "secret"])

    email, password = prompt_credentials(input_func=input_mock, password_func=password_mock)

    assert (email, password) == ("user@example.com", "secret")
    assert input_mock.call_count == 2  # lần đầu rỗng -> hỏi lại


def test_prompt_credentials_strip_email():
    email, password = prompt_credentials(
        input_func=lambda _: "  a@b.com  ", password_func=lambda _: "pw"
    )
    assert email == "a@b.com"
    assert password == "pw"
