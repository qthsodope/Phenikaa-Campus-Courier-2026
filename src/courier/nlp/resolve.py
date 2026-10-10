"""Giải tham chiếu không gian + tạo Mission cho khối 3.

Đã kiểm tra trên train (100% khớp rc của scenes.json):
  - bắc/nam/tây/đông = hàng nhỏ nhất / hàng lớn nhất / cột nhỏ nhất / cột lớn nhất (theo (hàng, cột));
  - gần/xa X = khoảng cách đường chim bay (Euclid) theo (hàng, cột) tới X.
"""
from ..graph import Mission


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def resolve_ref(ref, copies, landmarks):
    """ref = {kind, anchor}; copies = các bản của loại địa điểm; landmarks = {loại: [rc...]}.
    Trả về rc được chỉ, hoặc None nếu không xác định được (khi đó robot tự chọn)."""
    if ref and ref.get("rc") is not None:          # V2: nơi giao mô tả qua bản đồ -> ô đã được giải sẵn
        return tuple(ref["rc"])
    if not ref or len(copies) < 2:
        return None
    k = ref["kind"]
    if k == "north":
        return min(copies, key=lambda c: c[0])
    if k == "south":
        return max(copies, key=lambda c: c[0])
    if k == "west":
        return min(copies, key=lambda c: c[1])
    if k == "east":
        return max(copies, key=lambda c: c[1])
    anchors = landmarks.get(ref.get("anchor"), [])
    if not anchors:
        return None
    x = anchors[0]
    if k == "near":
        return min(copies, key=lambda c: _d2(c, x))
    if k == "far":
        return max(copies, key=lambda c: _d2(c, x))
    return None


def apply_constraints(parsed, landmarks):
    """Kiểm tra chéo với bản đồ (bước 5B), đúng 100% trên train:
      - goal / via phải là loại CÓ trên bản đồ; via khác goal;
      - tham chiếu chỉ có nghĩa khi loại đó có 2 bản; mốc "gần/xa X" phải là loại có ĐÚNG 1 bản.
    Chọn phương án xếp hạng cao nhất thỏa ràng buộc. Không có phương án nào thỏa thì giữ kết quả cũ."""
    from .mode import CONSTRAINTS
    c = parsed.get("_cands")
    if not CONSTRAINTS or not c or not landmarks:
        return parsed
    on = lambda t: t is not None and len(landmarks.get(t, [])) > 0  # noqa: E731
    out = dict(parsed)
    if "_probs" in parsed:                       # có mô hình: giải mã chung (xem model.py)
        from .model import joint_decode
        return joint_decode(parsed, landmarks)
    goal = next((t for t in c["goal"] if on(t)), parsed["goal"])
    via = next((t for t in c["via"] if on(t) and t != goal), None)
    if goal != parsed["goal"] or via != parsed["via"]:
        out["goal"], out["via"] = goal, via
        out["goal_ref"] = c["refs"].get(("goal", goal)) or c["refs"].get(("via", goal))
        out["via_ref"] = c["refs"].get(("via", via)) if via else None
    for f, t in (("goal_ref", out["goal"]), ("via_ref", out["via"])):
        r = out[f]
        if r and r.get("anchor") and len(landmarks.get(r["anchor"], [])) != 1:
            out[f] = None                        # mốc không hợp lệ -> bỏ tham chiếu (robot tự chọn)
    return out


def to_mission(parsed, landmarks):
    """Kết quả parse() + địa điểm trên bản đồ -> Mission (đúng loại khối 3 dùng)."""
    parsed = apply_constraints(parsed, landmarks)
    goal, via = parsed["goal"], parsed["via"]
    goal_rc = resolve_ref(parsed["goal_ref"], landmarks.get(goal, []), landmarks) if goal else None
    via_rc = resolve_ref(parsed["via_ref"], landmarks.get(via, []), landmarks) if via else None
    return Mission(goal=goal, via=via, goal_rc=goal_rc, via_rc=via_rc,
                   urgent=parsed["urgent"], fragile=parsed["fragile"])
