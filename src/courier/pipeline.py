"""Ghép 3 khối: ảnh (khối 1) + tin nhắn (khối 2) -> bản đồ + yêu cầu -> luật robot (khối 3) -> hướng đi."""
import json
from pathlib import Path

from .graph import Mission
from .nlp.parse import parse
from .nlp.resolve import to_mission
from .policy.v2 import predict_scene
from .vision.reader import VisionReader, scene_from_vision


def scene_groups(rows):
    """observations.json -> danh sách cảnh: (chỉ số dòng đầu, ảnh, câu yêu cầu). Mỗi cảnh = 10 dòng liên tiếp."""
    out = []
    for i in range(0, len(rows), 10):
        out.append((i, rows[i]["image"], rows[i]["mission"]))
    return out


def predict_scene_from_inputs(reader, image_path, text):
    """Một cảnh -> (10 hướng đi cho robot 0..9, kết quả đọc ảnh, kết quả đọc tin nhắn)."""
    v = reader.read(image_path)
    p = parse(text)
    if v is None:                                   # không đọc được ảnh: đoán "đi thẳng" (không bao giờ bỏ trống)
        return [0] * 10, None, p
    from .vision.repair import repair
    S = scene_from_vision(repair(v, p))             # H27: sửa bản đồ theo luật đề (COURIER_REPAIR, mặc định tắt)
    S.mission = to_mission(p, S.landmarks) if (p["goal"] or "_probs" in p) else Mission(goal=None)
    return predict_scene(S), v, p


def predict_rows(data_dir, split_rows, reader=None, progress=True):
    """Dự đoán cho mọi dòng của một bộ dữ liệu (theo đúng thứ tự). Trả về list số nguyên 0..3."""
    reader = reader or VisionReader()
    preds = [0] * len(split_rows)
    groups = scene_groups(split_rows)
    for k, (i0, image, text) in enumerate(groups):
        moves, _, _ = predict_scene_from_inputs(reader, Path(data_dir) / image, text)
        for r in range(10):
            assert split_rows[i0 + r]["robot_id"] == r
            preds[i0 + r] = int(moves[r])
        if progress and (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(groups)} cảnh", flush=True)
    return preds


def load_rows(split_dir):
    return json.loads((Path(split_dir) / "observations.json").read_text(encoding="utf-8"))
