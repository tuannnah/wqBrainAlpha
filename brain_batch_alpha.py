"""Module xử lý hàng loạt WorldQuant Brain API (adapter tương thích menu cũ).

Lớp `BrainBatchAlpha` kế thừa `WorldQuantClient` để dùng chung phần xác thực
và session. Các method sinh/simulate/submit theo template cũ được giữ lại cho
menu legacy cho tới khi Phase 4 chuyển toàn bộ caller sang pipeline mới.
"""

from datetime import datetime
from time import sleep

from alpha_strategy import AlphaStrategy
from dataset_config import get_dataset_config
from worldquant_client import AuthenticationError, WorldQuantClient

__all__ = ["AuthenticationError", "BrainBatchAlpha"]


class BrainBatchAlpha(WorldQuantClient):
    """Compatibility adapter cho menu legacy cho tới Phase 4."""

    def simulate_alphas(self, datafields=None, strategy_mode=1, dataset_name=None):
        """Mô phỏng danh sách Alpha."""

        try:
            datafields = self._get_datafields_if_none(datafields, dataset_name)
            if not datafields:
                return []

            alpha_list = self._generate_alpha_list(datafields, strategy_mode)
            if not alpha_list:
                return []

            print(f"\n🚀 Bắt đầu mô phỏng {len(alpha_list)} biểu thức Alpha...")

            results = []
            for i, alpha in enumerate(alpha_list, 1):
                print(f"\n[{i}/{len(alpha_list)}] Đang mô phỏng Alpha...")
                result = self._simulate_single_alpha(alpha)
                if result and result.get('passed_all_checks'):
                    results.append(result)
                    self._save_alpha_id(result['alpha_id'], result)

                if i < len(alpha_list):
                    sleep(5)

            return results

        except Exception as e:
            print(f"❌ Lỗi trong quá trình mô phỏng: {str(e)}")
            return []

    def _simulate_single_alpha(self, alpha):
        """Mô phỏng một Alpha."""

        try:
            print(f"Biểu thức: {alpha.get('regular', 'Unknown')}")

            # Gửi yêu cầu mô phỏng.
            sim_resp = self.session.post(
                f"{self.API_BASE_URL}/simulations",
                json=alpha
            )

            if sim_resp.status_code != 201:
                print(f"❌ Yêu cầu mô phỏng thất bại (mã trạng thái: {sim_resp.status_code})")
                return None

            try:
                sim_progress_url = sim_resp.headers['Location']
                start_time = datetime.now()
                total_wait = 0

                while True:
                    sim_progress_resp = self.session.get(sim_progress_url)
                    retry_after_sec = float(sim_progress_resp.headers.get("Retry-After", 0))

                    if retry_after_sec == 0:
                        alpha_id = sim_progress_resp.json()['alpha']
                        print(f"✅ Nhận được Alpha ID: {alpha_id}")

                        # Chờ thêm một chút để các chỉ số được tính xong.
                        sleep(3)

                        # Lấy chi tiết Alpha.
                        alpha_url = f"{self.API_BASE_URL}/alphas/{alpha_id}"
                        alpha_detail = self.session.get(alpha_url)
                        alpha_data = alpha_detail.json()

                        # API cần trả về field 'is' để đọc các chỉ số kiểm tra.
                        if 'is' not in alpha_data:
                            print("❌ Không lấy được dữ liệu chỉ số")
                            return None

                        is_qualified = self.check_alpha_qualification(alpha_data)

                        return {
                            'expression': alpha.get('regular'),
                            'alpha_id': alpha_id,
                            'passed_all_checks': is_qualified,
                            'metrics': alpha_data.get('is', {}),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                    total_wait += retry_after_sec
                    elapsed = (datetime.now() - start_time).total_seconds()
                    progress = min(95, (elapsed / 30) * 100)

                    print(f"⏳ Đang chờ kết quả mô phỏng... ({elapsed:.1f} giây | tiến độ khoảng {progress:.0f}%)")
                    sleep(retry_after_sec)

            except KeyError:
                print("❌ Không lấy được URL tiến độ mô phỏng")
                return None

        except Exception as e:
            print(f"⚠️ Mô phỏng Alpha thất bại: {str(e)}")
            return None

    def check_alpha_qualification(self, alpha_data):
        """Kiểm tra Alpha có đạt toàn bộ điều kiện submit hay không."""

        try:
            is_data = alpha_data.get('is', {})
            if not is_data:
                print("❌ Không lấy được dữ liệu chỉ số")
                return False

            sharpe = float(is_data.get('sharpe', 0))
            fitness = float(is_data.get('fitness', 0))
            turnover = float(is_data.get('turnover', 0))
            ic_mean = float(is_data.get('margin', 0))

            sub_universe_check = next(
                (
                    check for check in is_data.get('checks', [])
                    if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE'
                ),
                {}
            )
            subuniverse_sharpe = float(sub_universe_check.get('value', 0))
            required_subuniverse_sharpe = float(sub_universe_check.get('limit', 0))

            print("\n📊 Chi tiết chỉ số Alpha:")
            print(f"  Sharpe: {sharpe:.3f} (>1.5)")
            print(f"  Fitness: {fitness:.3f} (>1.0)")
            print(f"  Turnover: {turnover:.3f} (0.1-0.9)")
            print(f"  IC Mean: {ic_mean:.3f} (>0.02)")
            print(f"  Sub-universe Sharpe: {subuniverse_sharpe:.3f} (>{required_subuniverse_sharpe:.3f})")

            print("\n📝 Kết quả đánh giá chỉ số:")

            is_qualified = True

            if sharpe < 1.5:
                print("❌ Sharpe ratio chưa đạt chuẩn")
                is_qualified = False
            else:
                print("✅ Sharpe ratio đạt chuẩn")

            if fitness < 1.0:
                print("❌ Fitness chưa đạt chuẩn")
                is_qualified = False
            else:
                print("✅ Fitness đạt chuẩn")

            if turnover < 0.1 or turnover > 0.9:
                print("❌ Turnover nằm ngoài khoảng hợp lý")
                is_qualified = False
            else:
                print("✅ Turnover đạt chuẩn")

            if ic_mean < 0.02:
                print("❌ IC Mean chưa đạt chuẩn")
                is_qualified = False
            else:
                print("✅ IC Mean đạt chuẩn")

            if subuniverse_sharpe < required_subuniverse_sharpe:
                print(f"❌ Sub-universe Sharpe chưa đạt chuẩn ({subuniverse_sharpe:.3f} < {required_subuniverse_sharpe:.3f})")
                is_qualified = False
            else:
                print(f"✅ Sub-universe Sharpe đạt chuẩn ({subuniverse_sharpe:.3f} > {required_subuniverse_sharpe:.3f})")

            print("\n🔍 Kết quả từng kiểm tra:")
            checks = is_data.get('checks', [])
            for check in checks:
                name = check.get('name')
                result = check.get('result')
                value = check.get('value', 'N/A')
                limit = check.get('limit', 'N/A')

                if result == 'PASS':
                    print(f"✅ {name}: {value} (giới hạn: {limit})")
                elif result == 'FAIL':
                    print(f"❌ {name}: {value} (giới hạn: {limit})")
                    is_qualified = False
                elif result == 'PENDING':
                    print(f"⚠️ {name}: kiểm tra chưa hoàn tất")
                    is_qualified = False

            print("\n📋 Đánh giá cuối cùng:")
            if is_qualified:
                print("✅ Alpha đạt toàn bộ điều kiện, có thể submit!")
            else:
                print("❌ Alpha chưa đạt tiêu chuẩn submit")

            return is_qualified

        except Exception as e:
            print(f"❌ Lỗi khi kiểm tra điều kiện Alpha: {str(e)}")
            return False

    def _save_alpha_id(self, alpha_id, result, storage_path="alpha_ids.txt"):
        """Lưu Alpha ID đạt chuẩn vào file."""

        try:
            with open(storage_path, "a", encoding="utf-8") as f:
                f.write(f"{alpha_id}\n")
        except Exception as e:
            print(f"⚠️ Không lưu được Alpha ID: {str(e)}")

    def submit_alpha(self, alpha_id):
        """Submit một Alpha."""

        submit_url = f"{self.API_BASE_URL}/alphas/{alpha_id}/submit"

        for attempt in range(5):
            print(f"🔄 Lần thử submit Alpha {alpha_id}: {attempt + 1}")

            res = self.session.post(submit_url)
            if res.status_code == 201:
                print("✅ POST thành công, đang chờ quá trình submit hoàn tất...")
            elif res.status_code in [400, 403]:
                print(f"❌ Submit bị từ chối ({res.status_code})")
                return False
            else:
                sleep(3)
                continue

            while True:
                res = self.session.get(submit_url)
                retry = float(res.headers.get('Retry-After', 0))

                if retry == 0:
                    if res.status_code == 200:
                        print("✅ Submit thành công!")
                        return True
                    return False

                sleep(retry)

        return False

    def submit_multiple_alphas(self, alpha_ids):
        """Submit nhiều Alpha."""
        successful = []
        failed = []

        for alpha_id in alpha_ids:
            if self.submit_alpha(alpha_id):
                successful.append(alpha_id)
            else:
                failed.append(alpha_id)

            if alpha_id != alpha_ids[-1]:
                sleep(10)

        return successful, failed

    def _get_datafields_if_none(self, datafields=None, dataset_name=None):
        """Lấy danh sách data field nếu caller chưa truyền sẵn."""

        try:
            if datafields is not None:
                return datafields

            if dataset_name is None:
                print("❌ Chưa chỉ định dataset")
                return None

            config = get_dataset_config(dataset_name)
            if not config:
                print(f"❌ Dataset không hợp lệ: {dataset_name}")
                return None

            search_scope = {
                'instrumentType': 'EQUITY',
                'region': 'USA',
                'delay': '1',
                'universe': config['universe']
            }

            url_template = (
                f"{self.API_BASE_URL}/data-fields?"
                f"instrumentType={search_scope['instrumentType']}"
                f"&region={search_scope['region']}"
                f"&delay={search_scope['delay']}"
                f"&universe={search_scope['universe']}"
                f"&dataset.id={config['id']}"
                "&limit=50&offset={offset}"
            )

            initial_resp = self.session.get(url_template.format(offset=0))
            if initial_resp.status_code != 200:
                print("❌ Lấy data field thất bại")
                return None

            total_count = initial_resp.json()['count']

            all_fields = []
            for offset in range(0, total_count, 50):
                resp = self.session.get(url_template.format(offset=offset))
                if resp.status_code != 200:
                    continue
                all_fields.extend(resp.json()['results'])

            matrix_fields = [
                field['id'] for field in all_fields
                if field.get('type') == 'MATRIX'
            ]

            if not matrix_fields:
                print("❌ Không tìm thấy data field phù hợp")
                return None

            print(f"✅ Lấy được {len(matrix_fields)} data field")
            return matrix_fields

        except Exception as e:
            print(f"❌ Lỗi khi lấy data field: {str(e)}")
            return None

    def _generate_alpha_list(self, datafields, strategy_mode):
        """Sinh danh sách biểu thức Alpha."""
        try:
            strategy_generator = AlphaStrategy()

            strategies = strategy_generator.get_simulation_data(datafields, strategy_mode)

            print(f"Đã sinh {len(strategies)} biểu thức Alpha")

            alpha_list = []
            for strategy in strategies:
                simulation_data = {
                    'type': 'REGULAR',
                    'settings': {
                        'instrumentType': 'EQUITY',
                        'region': 'USA',
                        'universe': 'TOP3000',
                        'delay': 1,
                        'decay': 0,
                        'neutralization': 'SUBINDUSTRY',
                        'truncation': 0.08,
                        'pasteurization': 'ON',
                        'unitHandling': 'VERIFY',
                        'nanHandling': 'ON',
                        'language': 'FASTEXPR',
                        'visualization': False,
                    },
                    'regular': strategy
                }
                alpha_list.append(simulation_data)

            return alpha_list

        except Exception as e:
            print(f"❌ Sinh danh sách Alpha thất bại: {str(e)}")
            return []
