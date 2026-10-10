"""Đọc dữ liệu và chấm điểm giống BTC."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "delivery_v2"


def load_split(split, data=DATA):
    d = Path(data) / split
    rows = json.loads((d / "observations.json").read_text(encoding="utf-8"))
    lp = d / "labels.json"
    labels = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else None
    sp = d / "scenes.json"
    scenes = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None
    return rows, labels, scenes


def labels_by_scene(labels):
    """labels phẳng -> list mỗi cảnh 10 nhãn (robot 0..9)."""
    return [labels[i:i + 10] for i in range(0, len(labels), 10)]


def macro_accuracy(rows, labels, preds):
    per = defaultdict(list)
    for row, y, p in zip(rows, labels, preds):
        per[row["robot_id"]].append(y == p)
    scores = {r: sum(v) / len(v) for r, v in sorted(per.items())}
    return sum(scores.values()) / len(scores), scores
