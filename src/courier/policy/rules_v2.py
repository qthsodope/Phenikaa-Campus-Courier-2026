"""Luật tường minh của robot V2 (xem docs/luat_robot_v2.md)."""
import math

from .plan import first_move_costs, prepare
from .solver import turn_kind


def _r0_step(S, u, v, e, h, a, ph):
    """Robot 0 (Tia Chớp), chi phí so sánh theo bậc (train 100%, validation 100%):
    1) số đoạn; 2) số lần đi xuyên ô có địa điểm KHÔNG thuộc nhiệm vụ; 3) trời khô: số đoạn đông người,
    trời mưa: số đoạn KHÔNG có mái che (càng nhiều mái che càng tốt); 4) số đoạn đông người.
    Vẫn hòa: theo loại rẽ của bước đầu, thứ tự đổi theo gấp / mưa (r0_order)."""
    crowded = int(e.status == "crowded")
    third = -int(e.status == "covered") if S.rain else crowded
    return (1, int(v in S.lm), third, crowded)


# hòa nốt (so với hướng mũi; 0 thẳng, 1 phải, 2 trái, 3 quay đầu) — ĐỔI THEO TÌNH HUỐNG (khớp 100% ca hòa train + val):
#   không gấp: thẳng > phải > trái > quay đầu;  gấp + khô: PHẢI > thẳng > trái;  gấp + mưa: TRÁI > thẳng > phải
R0_ORDER = (0, 1, 2, 3)
R0_ORDER_URGENT_DRY = (1, 0, 2, 3)
R0_ORDER_URGENT_RAIN = (2, 0, 1, 3)


def r0_order(S):
    if not S.mission.urgent:
        return R0_ORDER
    return R0_ORDER_URGENT_RAIN if S.rain else R0_ORDER_URGENT_DRY


def robot0_move(S):
    if not hasattr(S, "lm"):
        prepare(S)
    if not S.goals:
        return S.heading
    c = first_move_costs(S, S.vias, S.goals, _r0_step)
    if not c:
        return S.heading
    best = min(c.values())
    opt = [a for a, x in c.items() if x == best]
    order = r0_order(S)
    return min(opt, key=lambda a: order.index(turn_kind(S.heading, a)))


# Robot 9 (Tham Lam) — chỉ nhìn các ô kề, so sánh theo bậc (nhỏ hơn = tốt hơn; dò 10/10, `logs/r9_unify.py`):
#   dễ vỡ: số đoạn đông · đêm: bước vào ô địa điểm · khoảng cách tới mục tiêu trước mắt (gấp: Manhattan, không gấp:
#   chim bay) · mưa: đoạn KHÔNG có mái che · ngày: bước vào ô địa điểm · không dễ vỡ: số đoạn đông.
#   Vẫn hòa: hướng TUYỆT ĐỐI — ngày Lên > Phải > Xuống > Trái; đêm Phải > Trái > Lên > Xuống.
#   Nhiều bản mục tiêu: chốt bản gần robot nhất (chim bay). Train 99,45%, validation 99,33%.
R9_ORDER_DAY = (0, 3, 1, 2)
R9_ORDER_NIGHT = (3, 2, 0, 1)


def _r9_key(S, b, e, T):
    m = S.mission
    k = []
    if m.fragile:
        k.append(int(e.status == "crowded"))
    if S.night:
        k.append(int(b in S.lm))
    if m.urgent:
        k.append(min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in T))
    else:
        k.append(round(min(math.dist(b, t) for t in T), 6))
    if S.rain:
        k.append(int(e.status != "covered"))
    if not S.night:
        k.append(int(b in S.lm))
    if not m.fragile:
        k.append(int(e.status == "crowded"))
    return tuple(k)


def robot9_move(S):
    if not hasattr(S, "lm"):
        prepare(S)
    T = S.vias or S.goals
    if not T:
        return S.heading
    # mục tiêu có nhiều bản: CHỐT bản gần robot nhất (chim bay) ngay từ đầu (train 1984 -> 1989; `logs/h2_copies.py`)
    T = [min(T, key=lambda t: (math.dist(S.robot, t), t))]
    order = R9_ORDER_NIGHT if S.night else R9_ORDER_DAY
    k = {a: _r9_key(S, b, e, T) + (order.index(a),) for a, b, e in S.adj[S.robot] if S.can_move(S.robot, b, e)}
    return min(k, key=k.get) if k else S.heading
