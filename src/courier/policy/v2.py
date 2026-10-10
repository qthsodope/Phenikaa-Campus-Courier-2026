"""Khối 3 V2: dự đoán bước đi của 10 robot.

- Robot 0: luật tường minh (rules_v2.robot0_move).
- Robot 9: luật tường minh (rules_v2.robot9_move).
- Robot 1-8: mô hình "chọn bước" (meta.MetaPolicy, models/policy_v2.pkl) — luật V2 của các robot này chưa giải mã
  hết thành lời (xem docs/luat_robot_v2.md); mô hình chọn trong các bước hợp lệ dựa trên một dàn bộ tìm đường
  và các bảng giá có điều kiện (cond.py: theo đêm / mưa / gấp / dễ vỡ / điểm ghé / loại nơi giao).
"""
import pickle
from pathlib import Path

from .cond import scene_paths
from .meta import build_rows, with_cond
from .rules_v2 import robot0_move, robot9_move

MODEL_PATH = Path(__import__("os").environ.get("COURIER_POLICY", "")
                  or Path(__file__).resolve().parents[3] / "models" / "policy_v2.pkl")
_POL = None
# 10/10: robot 9 dùng luật tường minh (rules_v2.robot9_move, train 99,2 / val 99,3) thay mô hình. 0 = mô hình cũ.
R9_RULE = int(__import__("os").environ.get("COURIER_R9_RULE", "1"))


def _policy():
    global _POL
    if _POL is None:
        _POL = pickle.loads(MODEL_PATH.read_bytes())
        from . import cond
        cond.DIJ = getattr(_POL, "dij", 0)      # bảng giá học với lộ trình Dijkstra -> dự đoán cũng dùng Dijkstra
        cond.R9RULE = getattr(_POL, "r9rule", 0)  # đặc trưng luật tường minh R9
        cond.XTAB = getattr(_POL, "xtab", 0)      # bảng giá số nguyên dò bằng tay
        cond.EXT = getattr(_POL, "ext", 0)        # đặc trưng v2 (thành phần lộ trình), 10/10
    return _POL


def predict_scene(S):
    """Scene (có mission) -> 10 hướng đi (0..3)."""
    if not S.mission or (S.mission.goal is None and S.mission.goal_rc is None):
        return [S.heading] * 10
    moves, X = build_rows(S)
    pol = _policy()
    X = with_cond(S, moves, X, scene_paths(S), pol.Ws)
    out = [robot0_move(S)]
    for r in range(1, 9):
        out.append(pol.predict(moves, X, r, S.heading))
    out.append(robot9_move(S) if R9_RULE else pol.predict(moves, X, 9, S.heading))
    return out
