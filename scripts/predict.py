"""MỘT LỆNH: thư mục dữ liệu (observations.json + images/) -> predictions.json. Chạy offline, không gọi dịch vụ ngoài.

    .venv/bin/python scripts/predict.py --data delivery_v2/test
    .venv/bin/python scripts/predict.py --data /duong/dan/private_test --out ket_qua.json

Mặc định ghi vào submissions/v2_<ngày giờ>.json (nên dùng --out tới /tmp; file nộp chính là submissions/BEST.json). Dùng được ngay cho bộ dữ liệu mới
(chỉ cần observations.json và thư mục ảnh; không cần scenes.json hay labels.json).
"""
import os

# CHẠY HOÀN TOÀN OFFLINE: không bao giờ gọi mạng / dịch vụ ngoài (đặt trước khi nạp transformers)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("COURIER_NLP_MODE", "final")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courier.pipeline import predict_rows, load_rows  # noqa: E402
from courier.vision.reader import VisionReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="thư mục chứa observations.json và images/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    a = ap.parse_args()

    need = ["kp.pt", "edge.pt", "node.pt", "misc.pt", "policy_v2.pkl", "nlp_final.pt", "nlp_aux_final.pkl",
            "phobert-base/config.json", "phobert-base/pytorch_model.bin"]
    missing = [n for n in need if not (a.models / n).exists()]
    if missing:
        sys.exit(f"thiếu file mô hình trong {a.models}: {missing} (xem README: huấn luyện lại bằng scripts/train_all_v2.sh)")
    # mã băm của mọi file mô hình được nạp -> in ra + ghi logs/predict_runs.log (biết file nộp do mô hình nào tạo)
    mode = os.environ["COURIER_NLP_MODE"]
    used = ["config.json", "kp.pt", "edge.pt", "node.pt", "misc.pt", "policy_v2.pkl", f"nlp_{mode}.pt", f"nlp_aux_{mode}.pkl"]
    files = sorted([a.models / n for n in used if (a.models / n).exists()] + [p for p in (a.models / "phobert-base").iterdir()])
    hashes = {str(p.relative_to(a.models)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for k, h in hashes.items():
        print(f"  mô hình {k}: sha256 {h[:16]}")
    rows = load_rows(a.data)
    t0 = time.time()
    reader = VisionReader(a.models)
    preds = predict_rows(a.data, rows, reader)
    dt = time.time() - t0

    assert len(preds) == len(rows) and all(isinstance(p, int) and 0 <= p <= 3 for p in preds), "sai định dạng đầu ra"
    out = a.out or ROOT / "submissions" / f"v2_{datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds), encoding="utf-8")
    print(f"đã ghi {len(preds)} dự đoán ({len(rows) // 10} cảnh) vào {out}")
    out_hash = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"sha256 file ra: {out_hash}")
    log = ROOT / "logs" / "predict_runs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}\tdata={a.data}\tout={out}\tsha256={out_hash}\tmodels={a.models}\t"
                + " ".join(f"{k}:{h[:16]}" for k, h in hashes.items()) + "\n")
    print(f"thời gian: {dt:.0f}s ({dt / max(len(rows) // 10, 1):.2f}s/cảnh)")


if __name__ == "__main__":
    main()
