"""Luật của 10 robot, suy ra từ train (xem README / báo cáo bước 2).

Mỗi robot (trừ 9) = tìm lộ trình rẻ nhất (qua điểm ghé rồi tới đích) theo "bảng giá" riêng.
Giá mặc định một đoạn đường = 1. Robot 9 chỉ nhìn một bước.
Hòa thì dùng chung thứ tự: thẳng > rẽ phải > rẽ trái > quay đầu (so với hướng mũi).
"""
from .solver import first_move_costs, pick, turn_kind

# ---------------------------------------------------------------- bảng giá
def _edge_cost(crowded=1.0, covered=1.0, normal=1.0, stairs_extra=0.0):
    table = {"normal": normal, "crowded": crowded, "covered": covered}

    def f(e):
        return table[e.status] + (stairs_extra if e.stairs else 0.0)
    return f


def _turn_cost(right, left, back):
    pen = (0.0, right, left, back)   # thẳng, phải, trái, quay đầu
    return lambda h, a: pen[turn_kind(h, a)]


def robot_config(rid, S):
    """Trả về (edge_cost, turn_cost, legged) của robot rid trong cảnh S."""
    m = S.mission
    if rid == 0:   # Tia Chớp: ít đoạn nhất
        return _edge_cost(), None, False
    if rid == 1:   # Yên Tĩnh: 1 đoạn đông người = 6 đoạn thường
        return _edge_cost(crowded=6), None, False
    if rid == 2:   # Mái Hiên: 1 đoạn mái che = nửa đoạn thường
        return _edge_cost(covered=0.5), None, False
    if rid == 3:   # Mây Mưa: mưa -> đường không mái che đắt gấp 5; khô -> đông người = 3
        if S.rain:
            return _edge_cost(covered=0.2), None, False
        return _edge_cost(crowded=3), None, False
    if rid == 4:   # Sơn Dương: đi được bậc thang; đông người = 2; mưa thì bậc thang +4
        return _edge_cost(crowded=2, stairs_extra=4.0 if S.rain else 0.0), None, True
    if rid == 5:   # Thẳng Tắp: mỗi lần rẽ +3, quay đầu +30
        return _edge_cost(), _turn_cost(3, 3, 30), False
    if rid == 6:   # Hỏa Tốc: gấp -> ít đoạn nhất; không gấp -> đông người = 5, mái che = 0.8
        if m.urgent:
            return _edge_cost(), None, False
        return _edge_cost(crowded=5, covered=0.8), None, False
    if rid == 7:   # Nâng Niu: dễ vỡ -> đông người = 7, rẽ +1.5, quay đầu +15; không -> ít đoạn nhất
        if m.fragile:
            return _edge_cost(crowded=7), _turn_cost(1.5, 1.5, 15), False
        return _edge_cost(), None, False
    if rid == 8:   # Lề Phải: rẽ phải +0.5, rẽ trái +4.5, quay đầu +9
        return _edge_cost(), _turn_cost(0.5, 4.5, 9), False
    raise ValueError(rid)


# ---------------------------------------------------------------- mục tiêu
def candidate_targets(S):
    """(danh sách bản điểm ghé hoặc None, danh sách bản đích)."""
    m = S.mission
    goals = [m.goal_rc] if m.goal_rc else list(S.landmarks.get(m.goal, []))
    vias = None
    if m.via:
        vias = [m.via_rc] if m.via_rc else list(S.landmarks.get(m.via, []))
        if not vias:
            vias = None
    return vias, goals


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


GREEDY_STEP = {"normal": 1.0, "covered": 1.0, "crowded": 2.0}


def greedy_move(S):
    """Tham Lam: mục tiêu trước mắt = điểm ghé (nếu có) hoặc đích; nhiều bản -> bản gần robot nhất
    (Manhattan). Chọn bước có (giá đoạn + Manhattan tới mục tiêu) nhỏ nhất."""
    vias, goals = candidate_targets(S)
    T = vias or goals
    if not T:
        return S.heading
    t = min(T, key=lambda x: _manhattan(S.robot, x))
    scores = {a: GREEDY_STEP[e.status] + _manhattan(b, t)
              for a, b, e in S.adj[S.robot] if S.can_move(S.robot, b, e)}
    return pick(scores, S.heading)


def predict(S, rid):
    if rid == 9:
        return greedy_move(S)
    vias, goals = candidate_targets(S)
    if not goals:
        return S.heading
    edge, turn, legged = robot_config(rid, S)
    return pick(first_move_costs(S, vias, goals, edge, turn, legged), S.heading)


def predict_scene(S):
    return [predict(S, r) for r in range(10)]
