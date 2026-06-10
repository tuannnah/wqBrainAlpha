import unittest
from unittest.mock import Mock, patch

import requests

from brain_batch_alpha import AuthenticationError, BrainBatchAlpha
from main import main, prompt_credentials


class ConsoleCredentialsTest(unittest.TestCase):
    def test_yeu_cau_nhap_lai_khi_email_hoac_mat_khau_rong(self):
        input_mock = Mock(side_effect=["", "user@example.com"])
        password_mock = Mock(side_effect=["", "secret"])

        email, password = prompt_credentials(
            input_func=input_mock,
            password_func=password_mock,
        )

        self.assertEqual((email, password), ("user@example.com", "secret"))
        self.assertEqual(input_mock.call_count, 2)
        self.assertEqual(password_mock.call_count, 2)


class MainEntryTest(unittest.TestCase):
    def test_main_dang_nhap_roi_goi_run_application(self):
        input_mock = Mock()

        with (
            patch("main.prompt_credentials", return_value=("user@example.com", "secret")),
            patch("main.run_application") as run_application,
            patch("builtins.print"),
        ):
            main(input_func=input_mock, pause_on_exit=False)

        run_application.assert_called_once_with(
            "user@example.com", "secret", input_func=input_mock
        )

    def test_giu_cua_so_mo_khi_gap_loi_sau_dang_nhap(self):
        input_mock = Mock(side_effect=[""])

        with (
            patch("main.prompt_credentials", return_value=("user@example.com", "secret")),
            patch("main.run_application", side_effect=RuntimeError("loi")),
            patch("builtins.print"),
        ):
            main(input_func=input_mock, pause_on_exit=True)

        input_mock.assert_any_call("\nNhấn Enter để đóng chương trình...")


class BrainAuthenticationTest(unittest.TestCase):
    def test_xac_thuc_thanh_cong_bang_tai_khoan_duoc_truyen_vao(self):
        response = Mock(status_code=201, headers={})
        response.json.return_value = {}
        session = Mock()
        session.post.return_value = response

        BrainBatchAlpha("user@example.com", "secret", session=session)

        self.assertEqual(session.auth.username, "user@example.com")
        self.assertEqual(session.auth.password, "secret")
        session.post.assert_called_once_with(
            "https://api.worldquantbrain.com/authentication",
            timeout=30,
        )

    def test_sai_tai_khoan_khong_lam_lo_thong_tin_nhay_cam(self):
        response = Mock(status_code=401, headers={})
        response.json.return_value = {}
        session = Mock()
        session.post.return_value = response

        with self.assertRaises(AuthenticationError) as context:
            BrainBatchAlpha("user@example.com", "secret", session=session)

        message = str(context.exception)
        self.assertIn("Email hoặc mật khẩu không đúng", message)
        self.assertNotIn("user@example.com", message)
        self.assertNotIn("secret", message)

    def test_loi_mang_duoc_chuyen_thanh_loi_xac_thuc(self):
        session = Mock()
        session.post.side_effect = requests.RequestException("network details")

        with self.assertRaises(AuthenticationError) as context:
            BrainBatchAlpha("user@example.com", "secret", session=session)

        message = str(context.exception)
        self.assertIn("Không thể kết nối đến WorldQuant BRAIN", message)
        self.assertNotIn("network details", message)


class BrainVerificationTest(unittest.TestCase):
    @staticmethod
    def _response(status_code, json_data=None, headers=None):
        response = Mock(status_code=status_code, headers=headers or {})
        response.json.return_value = json_data or {}
        return response

    def test_lay_duong_dan_xac_thuc_tu_header_hoac_json(self):
        cases = [
            (
                self._response(
                    202,
                    headers={"Location": "https://platform.worldquantbrain.com/verify"},
                ),
                "https://platform.worldquantbrain.com/verify",
            ),
            (
                self._response(401, {"inquiryUrl": "/verify/inquiry-123"}),
                "https://api.worldquantbrain.com/verify/inquiry-123",
            ),
            (
                self._response(
                    401,
                    {"challenge": {"verificationUrl": "https://example.com/qr"}},
                ),
                "https://example.com/qr",
            ),
        ]

        for response, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    BrainBatchAlpha._extract_verification_url(response),
                    expected,
                )

    def test_mo_trinh_duyet_va_kiem_tra_lai_sau_khi_nguoi_dung_xac_nhan(self):
        challenge = self._response(
            401,
            {"inquiryUrl": "https://platform.worldquantbrain.com/verify/abc"},
        )
        success = self._response(201)
        session = Mock()
        session.post.side_effect = [challenge, success]
        browser_open = Mock(return_value=True)
        confirmation_input = Mock(return_value="")

        with patch("builtins.print") as print_mock:
            BrainBatchAlpha(
                "user@example.com",
                "secret",
                session=session,
                browser_open=browser_open,
                confirmation_input=confirmation_input,
            )

        browser_open.assert_called_once_with(
            "https://platform.worldquantbrain.com/verify/abc"
        )
        confirmation_input.assert_called_once()
        self.assertEqual(session.post.call_count, 2)
        self.assertTrue(
            any(
                "https://platform.worldquantbrain.com/verify/abc" in str(call)
                for call in print_mock.call_args_list
            )
        )

    def test_in_duong_dan_khi_trinh_duyet_khong_mo_duoc(self):
        challenge = self._response(
            202,
            {"verification_url": "https://example.com/manual"},
        )
        success = self._response(200)
        session = Mock()
        session.post.side_effect = [challenge, success]
        browser_open = Mock(side_effect=OSError("cannot open"))

        with patch("builtins.print") as print_mock:
            BrainBatchAlpha(
                "user@example.com",
                "secret",
                session=session,
                browser_open=browser_open,
                confirmation_input=Mock(return_value=""),
            )

        self.assertTrue(
            any(
                "Hãy mở đường dẫn phía trên thủ công" in str(call)
                for call in print_mock.call_args_list
            )
        )

    def test_bao_loi_khi_challenge_khong_co_duong_dan(self):
        session = Mock()
        session.post.return_value = self._response(202)

        with self.assertRaises(AuthenticationError) as context:
            BrainBatchAlpha("user@example.com", "secret", session=session)

        self.assertIn("không trả về đường dẫn xác thực", str(context.exception))

    def test_dung_sau_ba_lan_xac_thuc_chua_hoan_tat(self):
        challenge = self._response(
            401,
            {"inquiryUrl": "https://example.com/verify"},
        )
        session = Mock()
        session.post.side_effect = [challenge, challenge, challenge, challenge]
        confirmation_input = Mock(return_value="")

        with self.assertRaises(AuthenticationError) as context:
            BrainBatchAlpha(
                "user@example.com",
                "secret",
                session=session,
                browser_open=Mock(return_value=True),
                confirmation_input=confirmation_input,
            )

        self.assertIn("chưa hoàn tất", str(context.exception))
        self.assertEqual(confirmation_input.call_count, 3)
        self.assertEqual(session.post.call_count, 4)


if __name__ == "__main__":
    unittest.main()
