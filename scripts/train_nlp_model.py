"""Huấn luyện mô hình ngôn ngữ của khối 2 (bước 5B). Chế độ lấy từ COURIER_NLP_MODE (blind | final).

    # dev nội bộ (so sánh bộ mã hóa, KHÔNG dùng validation): học 80% cảnh train + nửa A kho cụm từ/khung câu,
    # đo trên 20% cảnh train còn lại biến đổi bằng nửa B (cụm từ + khung câu chưa thấy)
    COURIER_NLP_MODE=blind .venv/bin/python scripts/train_nlp_model.py --backbone visobert --dev
    # mô hình thật: chế độ mù = mọi cảnh train; chế độ cuối = train + toàn bộ validation -> models/nlp_<mode>.pt
    COURIER_NLP_MODE=blind .venv/bin/python scripts/train_nlp_model.py --backbone visobert
"""
import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from courier.data import load_split  # noqa: E402
from courier.nlp import mode  # noqa: E402
from courier.nlp.augment import build_dataset  # noqa: E402
from courier.nlp.model import HEADS, encode_label, n_params, fit, MODELS, model_path  # noqa: E402

SEED = 2026


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(net, tok, data, device, bs=128):
    net.eval()
    ok, full = Counter(), 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            X, M = tok([t for t, _ in chunk])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = net(X.to(device), M.to(device))
            pred = {k: v.argmax(-1).cpu().tolist() for k, v in out.items()}
            for j, (_, lab) in enumerate(chunk):
                y = encode_label(lab)
                good = {k: pred[k][j] == y[k] for k in HEADS if y[k] != -100}
                for k, g in good.items():
                    ok[k] += g
                full += all(good.values())
    n = len(data)
    net.train()
    return {k: ok[k] / n for k in HEADS}, full / n   # (goal bị bỏ qua ở câu mô tả bản đồ: tỷ lệ goal hơi thấp hơn thực)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="visobert", choices=["visobert", "phobert", "e5"])
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--n-synth", type=int, default=15000)
    a = ap.parse_args()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("cấu hình:", mode.describe(), "| bộ mã hóa:", a.backbone, "| dev nội bộ:" , a.dev, flush=True)
    _, _, tr = load_split("train")
    if mode.MODE == "final" and not a.dev:
        tr = tr + mode.final_val_scenes()
    t0 = time.time()
    if a.dev:
        idx = list(range(len(tr)))
        random.Random(SEED).shuffle(idx)
        cut = int(0.8 * len(idx))
        fit_sc, dev_sc = [tr[i] for i in idx[:cut]], [tr[i] for i in idx[cut:]]
        train = build_dataset(fit_sc, a.n_real, a.n_synth, pool="A", seed=SEED)
        devB = build_dataset(dev_sc, n_real=3, n_synth=2000, pool="B", seed=7)
        n0 = len(dev_sc)
        dev_sets = {"câu gốc giữ lại": devB[:0] + [d for k, d in enumerate(devB[:n0 * 4]) if k % 4 == 0],
                    "câu thật biến đổi (nửa B)": [d for k, d in enumerate(devB[:n0 * 4]) if k % 4 != 0],
                    "câu tổng hợp (khung B)": devB[n0 * 4:]}
    else:
        train = build_dataset(tr, a.n_real, a.n_synth, pool=None, seed=SEED)
        dev_sets = {}
    print(f"dữ liệu học: {len(train)} câu (sinh trong {time.time() - t0:.0f} giây)", flush=True)
    net, tok, device = fit(train, a.backbone, epochs=a.epochs, bs=a.bs, lr=a.lr, seed=SEED,
                           log=lambda m: print(m, flush=True))
    print(f"số tham số: {n_params(net) / 1e6:.2f} triệu", flush=True)
    for name, ds in dev_sets.items():
        acc, full = evaluate(net, tok, ds, device)
        print(f"  [dev] {name} ({len(ds)}): trọn câu {full * 100:.2f}% | "
              + " ".join(f"{k} {v * 100:.1f}" for k, v in acc.items()), flush=True)
    if not a.dev:
        MODELS.mkdir(exist_ok=True)
        torch.save({"backbone": a.backbone, "state": net.state_dict(), "mode": mode.MODE}, model_path())
        print("đã lưu", model_path())
    print(f"xong trong {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
