#!/usr/bin/env bash
# Chạy luồng 5 (closed-loop) lặp nhiều lần — mỗi lần seed ngẫu nhiên mới (~6 ý tưởng/lần)
# để tích lũy nhiều lượt thử Brain sim. Tự dừng nếu 1 lần thất bại (vd session hết hạn -> QR).
cd "$(dirname "$0")"
for i in $(seq 1 30); do
  echo "===== LAN $i | $(date '+%H:%M:%S') ====="
  venv/Scripts/python.exe -u main.py closed-loop --market-data-dir data/market_yf >> wq_loop5.log 2>&1
  rc=$?
  echo "----- LAN $i xong (exit=$rc) | $(date '+%H:%M:%S') -----"
  if [ $rc -ne 0 ]; then echo "!! Dung lap: exit $rc (co the session het han)"; break; fi
done
echo "===== KET THUC LAP | $(date '+%H:%M:%S') ====="
