"""Bộ giải chung: tìm lộ trình rẻ nhất theo "bảng giá" của từng robot, trả về bước đi đầu tiên.

Trạng thái = (giao lộ, hướng đang đi, đã ghé điểm via chưa). Nhờ giữ hướng đi trong trạng thái,
bảng giá có thể tính cả chi phí rẽ (robot 5, 8).
"""
import heapq

from ..graph import DELTA

INF = float("inf")
EPS = 1e-9

# Phá hòa chung: thẳng > rẽ phải > rẽ trái > quay đầu (so với hướng mũi robot).
RIGHT_OF = {0: 3, 3: 1, 1: 2, 2: 0}   # UP->RIGHT, RIGHT->DOWN, DOWN->LEFT, LEFT->UP
LEFT_OF = {v: k for k, v in RIGHT_OF.items()}
BACK_OF = {0: 1, 1: 0, 2: 3, 3: 2}


def turn_kind(h, a):
    """0 = thẳng, 1 = rẽ phải, 2 = rẽ trái, 3 = quay đầu."""
    if a == h:
        return 0
    if a == RIGHT_OF[h]:
        return 1
    if a == LEFT_OF[h]:
        return 2
    return 3


def tie_order(heading):
    return sorted(range(4), key=lambda a: turn_kind(heading, a))


def route_cost(S, start, heading, targets_via, targets_goal, edge_cost, turn_cost=None, legged=False):
    """Chi phí rẻ nhất từ (start, heading) qua một điểm trong targets_via (nếu có) rồi tới một điểm
    trong targets_goal. Trả về INF nếu không tới được."""
    phase0 = 0 if targets_via else 1
    via = set(targets_via or ())
    goal = set(targets_goal)
    st = (start, heading, 1 if (phase0 == 0 and start in via) else phase0)
    d = {st: 0.0}
    pq = [(0.0, st)]
    while pq:
        c, (n, h, ph) = heapq.heappop(pq)
        if c > d[(n, h, ph)] + EPS:
            continue
        if ph == 1 and n in goal:
            return c
        for a, b, e in S.adj[n]:
            if not S.can_move(n, b, e, legged):
                continue
            nc = c + edge_cost(e) + (turn_cost(h, a) if turn_cost else 0.0)
            nph = 1 if (ph == 1 or b in via) else 0
            k = (b, a, nph)
            if nc < d.get(k, INF) - EPS:
                d[k] = nc
                heapq.heappush(pq, (nc, k))
    return INF


def first_move_costs(S, targets_via, targets_goal, edge_cost, turn_cost=None, legged=False):
    """Với mỗi bước đi đầu hợp lệ: chi phí cả lộ trình nếu đi bước đó."""
    out = {}
    for a, b, e in S.adj[S.robot]:
        if not S.can_move(S.robot, b, e, legged):
            continue
        c0 = edge_cost(e) + (turn_cost(S.heading, a) if turn_cost else 0.0)
        if targets_via is None and b in targets_goal:
            out[a] = c0
            continue
        via = targets_via
        if via is not None and b in via:
            via = None  # đã ghé xong
        out[a] = c0 + route_cost(S, b, a, via, targets_goal, edge_cost, turn_cost, legged)
    return out


def pick(costs, heading):
    """Chọn bước rẻ nhất; hòa thì theo thứ tự thẳng > phải > trái > quay đầu."""
    if not costs:
        return heading
    best = min(costs.values())
    for a in tie_order(heading):
        if a in costs and costs[a] <= best + EPS:
            return a
    return heading
