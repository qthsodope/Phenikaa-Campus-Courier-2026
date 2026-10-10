"""Bộ tìm đường tổng quát cho V2: chi phí mỗi bước = phí đoạn + phí rẽ + phí ô đi vào.

Trạng thái = (giao lộ, hướng đang đi, đã ghé điểm via chưa). Chi phí có thể là số thực hoặc TUPLE (so sánh từ điển:
bậc 1 quan trọng nhất, bậc sau chỉ dùng để phá hòa) — dùng cho luật robot 0.
"""
import heapq

from .solver import turn_kind  # noqa: F401  (dùng lại định nghĩa thẳng / phải / trái / quay đầu)


def _add(x, y):
    if isinstance(x, tuple):
        return tuple(p + q for p, q in zip(x, y))
    return x + y


def first_move_costs(S, vias, goals, step_cost, legged=False):
    """step_cost(S, u, v, e, h, a, phase) -> chi phí (số hoặc tuple) của bước u->v theo hướng a, đang hướng h.
    Trả {bước đầu a: chi phí rẻ nhất của cả lộ trình (qua via rồi tới goal) nếu đi bước đó}."""
    goals = set(goals)
    vias = set(vias) if vias else None
    out = {}
    for a, b, e in S.adj[S.robot]:
        if not S.can_move(S.robot, b, e, legged):
            continue
        ph = 1 if (vias is None or b in vias) else 0
        c0 = step_cost(S, S.robot, b, e, S.heading, a, ph)
        if ph == 1 and b in goals:
            out[a] = c0
            continue
        st = (b, a, ph)
        d = {st: c0}
        pq = [(c0, st)]
        while pq:
            c, k = heapq.heappop(pq)
            if c > d[k]:
                continue
            n, h, p = k
            if p == 1 and n in goals:
                out[a] = c
                break
            for a2, b2, e2 in S.adj[n]:
                if not S.can_move(n, b2, e2, legged):
                    continue
                np_ = 1 if (p == 1 or (vias is not None and b2 in vias)) else 0
                nc = _add(c, step_cost(S, n, b2, e2, h, a2, np_))
                k2 = (b2, a2, np_)
                if k2 not in d or nc < d[k2]:
                    d[k2] = nc
                    heapq.heappush(pq, (nc, k2))
    return out


def weighted(normal=1.0, crowded=1.0, covered=1.0, stairs=0.0, oneway=0.0, landmark=0.0, right=0.0, left=0.0,
             back=0.0):
    """Bảng giá cộng tuyến tính (một số thực mỗi bước)."""
    st = {"normal": normal, "crowded": crowded, "covered": covered}
    turn = (0.0, right, left, back)

    def f(S, u, v, e, h, a, ph):
        return (st[e.status] + (stairs if e.stairs else 0.0) + (oneway if e.oneway_to is not None else 0.0)
                + (landmark if v in S.lm else 0.0) + turn[turn_kind(h, a)])
    return f


def prepare(S):
    """Gắn vào cảnh: danh sách bản điểm ghé / đích (S.vias, S.goals) và tập ô địa điểm KHÔNG thuộc nhiệm vụ (S.lm)."""
    m = S.mission
    goals = [m.goal_rc] if m.goal_rc else list(S.landmarks.get(m.goal, []))
    vias = None
    if m.via:
        vias = [m.via_rc] if m.via_rc else list(S.landmarks.get(m.via, []))
        vias = vias or None
    S.goals, S.vias = goals, vias
    lm = set()
    for xs in S.landmarks.values():
        lm |= set(xs)
    S.lm = lm - set(goals) - set(vias or [])
    return S
