"""Bảng giá CÓ ĐIỀU KIỆN cho robot 1–8 (V2).

Phát hiện trên train V2: mỗi robot đổi bảng giá theo tình huống — ảnh ban đêm (kiểu vẽ "night"), mưa, gấp, dễ vỡ,
có điểm ghé, và LOẠI NƠI GIAO (vd Mái Hiên giao tới giảng đường / phòng thí nghiệm / thư viện thì rất né đường đông;
Mây Mưa khi đó lại né rẽ như Thẳng Tắp). Một bảng giá chung chỉ khớp 75–86%; bảng giá có điều kiện khớp 81–92% (val-A).

Mô hình: chi phí một lộ trình = tổng theo từng bước của  w(tình huống) · f(bước), với
  f(bước) = [1 đoạn, đông, mái che, bậc thang, một chiều, đi vào ô địa điểm KHÔNG thuộc nhiệm vụ,
             rẽ phải, rẽ trái, quay đầu, (rẽ phải, rẽ trái, quay đầu) ở riêng bước đầu]
  w(tình huống) = W @ c,  c = [1, đêm, mưa, gấp, dễ vỡ, có điểm ghé, loại nơi giao (10 ô một-nóng)].
W học bằng "softmin + entropy chéo" trên các lộ trình liệt kê sẵn (độ dài <= ngắn nhất + EXTRA, không lặp ô).
Dùng làm ĐẶC TRƯNG cho mô hình chọn bước (meta.py): mỗi bước đắt hơn bước rẻ nhất bao nhiêu theo bảng giá của robot r.
"""
import math
import os

import numpy as np

from .plan import first_move_costs, prepare
from .solver import turn_kind

GOALS = ["canteen", "clinic", "dorm", "gate", "lab", "lecture", "library", "office", "parking", "sports"]
NF = 12
EXTRA = 4
EXTRA2 = int(os.environ.get("COURIER_EXTRA2", "0"))      # 0 = tắt. Hướng đi đầu chưa có lộ trình nào trong (ngắn nhất + EXTRA): liệt kê riêng tới ngắn nhất + EXTRA2
# 09/10: bổ sung lộ trình tối ưu thật (Dijkstra trên trạng thái ô + hướng, ĐƯỢC đi lại ô cũ: "đi ra 1 ô rồi quay lại" để
# né quay đầu tại chỗ) vào tập lộ trình liệt kê, cả khi học (DIJ vòng) lẫn khi dự đoán. 0 = tắt (như bản BEST 08/10).
DIJ = int(os.environ.get("COURIER_DIJ", "0"))
# 09/10: luật tường minh của Tham Lam (R9) làm đặc trưng (0 = tắt). Khóa so sánh từng bước kề, nhỏ hơn = tốt hơn:
#   dễ vỡ: số đoạn đông · đêm: vào ô địa điểm · khoảng cách tới đích (gấp: theo ô / Manhattan; không gấp: chim bay)
#   · ngày: vào ô địa điểm · không dễ vỡ: đoạn đông · hòa: Lên > Phải > Xuống > Trái. Train 95,45%. R9RULE=1: 2 cột; 2: 3 cột.
R9RULE = int(os.environ.get("COURIER_R9RULE", "0"))
# 09/10: bảng giá số nguyên dò bằng tay (Dijkstra chính xác, nhóm cảnh cơ bản) làm "cố vấn" thêm (0 = tắt).
# thứ tự đặc trưng: đoạn, đông, mái, bậc, một chiều, địa điểm, rẽ phải, rẽ trái, quay đầu, (bước đầu) phải, trái, quay đầu
XTAB = int(os.environ.get("COURIER_XTAB", "0"))
# 10/10: "đặc trưng v2" (theo bài khác) — thành phần lộ trình phía sau mỗi bước (12 cột NF) theo lộ trình rẻ nhất của bảng
# giá từng robot 1-8 + lộ trình ít đoạn nhất -> 108 cột nối CUỐI ma trận. Chỉ robot có kiểu "x…" (meta.py) dùng. 0 = tắt.
EXT = int(os.environ.get("COURIER_EXT", "0"))
EXT_COLS = 9 * 12
XTABLES = [
    np.array([1, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], float),    # R1: đông = đi xuyên địa điểm = +4 (193/199)
    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], float),    # R6 gấp / đêm: ít đoạn + quay đầu ngay +2 (96,8% trong nhóm tối ưu)
    np.array([1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2], float),    # R6 ngày không gấp (174/199)
    np.array([1, 1, 0, 0, 0, 1, 0, 3, 6, 0, 0, 1], float),    # R8: rẽ trái +3, quay đầu +6 (188/199)
]
CAP = 20000
CAP2 = 3000
BIG = 50.0


def cvec(S):
    m = S.mission
    return np.array([1.0, S.night, S.rain, m.urgent, m.fragile, m.via is not None] + [m.goal == g for g in GOALS], float)


def enum_paths(S, legged=False):
    """Mọi lộ trình tới nơi giao (qua điểm ghé nếu có), độ dài <= ngắn nhất + EXTRA, không lặp (ô, giai đoạn).
    -> ma trận (số lộ trình khác nhau, NF + 1): đặc trưng cộng dồn + bước đầu (cột cuối). None nếu không có."""
    if not hasattr(S, "lm"):
        prepare(S)
    if not S.goals:
        return None
    sp = first_move_costs(S, S.vias, S.goals, lambda S_, u, v, e, h, a, p: 1, legged)
    if not sp:
        return None
    L = min(sp.values()) + EXTRA
    goals = set(S.goals)
    vias = set(S.vias) if S.vias else None
    out = []

    lim = [L, CAP]

    def dfs(n, h, ph, f, depth, seen, only=None):
        if len(out) > lim[1]:
            return
        if ph == 1 and n in goals:
            out.append(f.copy())
            return
        if depth >= lim[0]:
            return
        for a, b, e in S.adj[n]:
            if depth == 0 and only is not None and a != only:
                continue
            if not S.can_move(n, b, e, legged):
                continue
            nph = 1 if (ph == 1 or (vias is not None and b in vias)) else 0
            key = (b, nph)
            if key in seen:
                continue
            tk = turn_kind(h, a)
            g = f.copy()
            g[0] += 1
            g[1] += e.status == "crowded"
            g[2] += e.status == "covered"
            g[3] += bool(e.stairs)
            g[4] += bool(e.oneway_to)
            g[5] += b in S.lm
            if tk:
                g[5 + tk] += 1
                if depth == 0:
                    g[8 + tk] += 1
            if depth == 0:
                g[NF] = a
            seen.add(key)
            dfs(b, a, nph, g, depth + 1, seen)
            seen.discard(key)

    ph0 = 1 if (vias is None or S.robot in vias) else 0
    dfs(S.robot, S.heading, ph0, np.zeros(NF + 1), 0, {(S.robot, ph0)})
    # hướng đi đầu hợp lệ nhưng chưa có lộ trình (robot ghét rẽ / quay đầu có thể đi vòng rất xa)
    have = {int(x[NF]) for x in out}
    for a, b, e in S.adj[S.robot]:
        if EXTRA2 and a in sp and a not in have:
            n0 = len(out)
            lim[0], lim[1] = min(sp.values()) + EXTRA2, n0 + CAP2
            dfs(S.robot, S.heading, ph0, np.zeros(NF + 1), 0, {(S.robot, ph0)}, only=a)
    return np.unique(np.array(out), axis=0) if out else None


def scene_paths(S):
    """(lộ trình không chân, lộ trình có chân, đặc trưng tham lam của R9) của một cảnh."""
    return enum_paths(S, False), enum_paths(S, True), greedy_feats(S)


def move_costs(M, w):
    """M: ma trận lộ trình, w: trọng số (NF,) -> {bước đầu: chi phí nhỏ nhất}."""
    if M is None:
        return {}
    c = M[:, :NF] @ w
    out = {}
    for a, x in zip(M[:, NF].astype(int), c):
        if x < out.get(a, np.inf):
            out[a] = x
    return out


def dij_paths(S, w, legged=False):
    """Với mỗi bước đầu hợp lệ: lộ trình rẻ nhất theo bảng giá w (Dijkstra, trạng thái = ô, hướng, đã ghé chưa; mỗi bước
    tốn ít nhất 0,01). -> ma trận (số bước đầu, NF + 1) cùng dạng enum_paths, hoặc None."""
    import heapq
    if not hasattr(S, "lm"):
        prepare(S)
    if not S.goals:
        return None
    goal = set(S.goals)
    via = set(S.vias) if S.vias else None

    def feat(h, a, b, e, first):
        f = np.zeros(NF)
        f[0] = 1
        f[1] = e.status == "crowded"
        f[2] = e.status == "covered"
        f[3] = bool(e.stairs)
        f[4] = bool(e.oneway_to)
        f[5] = b in S.lm
        tk = turn_kind(h, a)
        if tk:
            f[5 + tk] = 1
            if first:
                f[8 + tk] = 1
        return f

    out = []
    for a, b, e in S.adj[S.robot]:
        if not S.can_move(S.robot, b, e, legged):
            continue
        ph = 1 if (via is None or b in via) else 0
        f0 = feat(S.heading, a, b, e, True)
        st = (b, a, ph)
        d, F, pq, end, cnt = {st: max(float(f0 @ w), 0.01)}, {st: f0}, [], None, 0
        pq.append((d[st], 0, st))
        while pq:
            c, _, k = heapq.heappop(pq)
            if c > d[k] + 1e-9:
                continue
            n, h, p = k
            if p == 1 and n in goal:
                end = k
                break
            for a2, b2, e2 in S.adj[n]:
                if not S.can_move(n, b2, e2, legged):
                    continue
                f = feat(h, a2, b2, e2, False)
                nc = c + max(float(f @ w), 0.01)
                k2 = (b2, a2, 1 if (p == 1 or (via is not None and b2 in via)) else 0)
                if nc < d.get(k2, np.inf) - 1e-9:
                    d[k2], F[k2] = nc, F[k] + f
                    cnt += 1
                    heapq.heappush(pq, (nc, cnt, k2))
        if end is not None:
            out.append(np.append(F[end], a))
    return np.array(out) if out else None


def merge_paths(M, N):
    if N is None:
        return M
    if M is None:
        return N
    return np.unique(np.vstack([M, N]), axis=0)


def gaps(M, w, moves):
    c = move_costs(M, w)
    if not c:
        return {a: BIG for a in moves}
    b = min(c.values())
    return {a: min(c[a] - b, BIG) if a in c else BIG for a in moves}


def weights(W, c):
    """Bảng giá cho tình huống c. W: ma trận (cộng theo điều kiện) hoặc dict mạng nhỏ (kết hợp điều kiện)."""
    if isinstance(W, dict):
        return W["W"] @ c + W["B"] @ np.tanh(W["A"] @ c + W["a"]) + W["b"]
    return W @ c


def fit_dij(data, rounds=None, legged=False, **kw):
    """Như fit, nhưng data = list (M, c, nhãn, S): sau mỗi lần học, bổ sung lộ trình Dijkstra theo bảng giá vừa học rồi
    học lại (DIJ vòng)."""
    rounds = DIJ if rounds is None else rounds
    Ms = [M for M, _, _, _ in data]
    for rd in range(rounds + 1):
        W = fit([(M, c, y) for M, (_, c, y, _) in zip(Ms, data)], **kw)
        if rd == rounds:
            return W
        Ms = [merge_paths(M, dij_paths(S, weights(W, c), legged)) for M, (_, c, _, S) in zip(Ms, data)]


def fit(data, epochs=60, l1=1e-3, lr=0.03, seed=2026, hidden=0):
    """data: list (M, c, nhãn). -> W (NF, len(c)), hoặc (hidden > 0) dict mạng nhỏ c -> bảng giá
    (phần cộng theo điều kiện + phần kết hợp điều kiện qua 1 lớp ẩn tanh). Softmin-CE trên chi phí nhỏ nhất của
    từng bước đầu (tất định)."""
    import torch
    nt = torch.get_num_threads()
    torch.set_num_threads(1)             # tất định: cùng kết quả mọi máy / mọi số luồng
    torch.manual_seed(seed)
    D = [(torch.tensor(M[:, :NF], dtype=torch.float32), torch.tensor(M[:, NF].astype(np.int64)),
          torch.tensor(c, dtype=torch.float32), int(y)) for M, c, y in data if M is not None and y in set(M[:, NF].astype(int))]
    nc = len(data[0][1])
    W = torch.zeros(NF, nc)
    W[0, 0] = 1.0
    W.requires_grad_(True)
    params = [W]
    if hidden:
        lin_a, lin_b = torch.nn.Linear(nc, hidden), torch.nn.Linear(hidden, NF)
        torch.nn.init.zeros_(lin_b.weight)
        torch.nn.init.zeros_(lin_b.bias)
        params += list(lin_a.parameters()) + list(lin_b.parameters())
    wfun = (lambda c: W @ c + lin_b(torch.tanh(lin_a(c)))) if hidden else (lambda c: W @ c)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(D))
        for b in range(0, len(D), 64):
            loss = 0
            for j in perm[b:b + 64]:
                Mt, mv, c, y = D[j]
                mc = torch.full((4,), float("inf")).scatter_reduce(0, mv, Mt @ wfun(c), "amin")
                logits = torch.where(torch.isinf(mc), torch.tensor(-1e4), -mc * 4)
                loss = loss + torch.nn.functional.cross_entropy(logits[None], torch.tensor([y]))
            loss = loss / 64 + l1 * W.abs().sum()
            if hidden:
                loss = loss + l1 * sum(p.abs().sum() for p in lin_b.parameters())
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.set_num_threads(nt)
    if hidden:
        g = lambda t: t.detach().numpy().astype(np.float64)  # noqa: E731
        return {"W": g(W), "A": g(lin_a.weight), "a": g(lin_a.bias), "B": g(lin_b.weight), "b": g(lin_b.bias)}
    return W.detach().numpy().astype(np.float64)


# ---------------------------------------------------------------- robot 9 (Tham Lam): chỉ nhìn một bước
# Điểm của mỗi bước kề = (W9 @ c) · g(bước), g = [Manhattan, chim bay, số đoạn ngắn nhất (cả nhiệm vụ), số đoạn ngắn nhất tới
# mục tiêu trước mắt, đông, mái che, đi vào địa điểm, rẽ phải, rẽ trái, quay đầu, tới ngay mục tiêu, một chiều, số lối ra].
# Vd dễ vỡ: né đông trước; không dễ vỡ: lại gần đích (chim bay) trước; đêm: né địa điểm. val-A ~89% (cũ 81%).
NG = 13


def greedy_feats(S):
    """-> (bước hợp lệ, ma trận g) cho robot không chân."""
    if not hasattr(S, "lm"):
        prepare(S)
    T = S.vias or S.goals
    if not T:
        return [], np.zeros((0, NG))
    one = lambda S_, u, v, e, h, a, p: 1  # noqa: E731
    bfs = first_move_costs(S, S.vias, S.goals, one)
    bfs1 = first_move_costs(S, None, T, one)
    mv, X = [], []
    for a, b, e in S.adj[S.robot]:
        if not S.can_move(S.robot, b, e):
            continue
        tk = turn_kind(S.heading, a)
        mv.append(a)
        X.append([min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in T), min(math.dist(b, t) for t in T),
                  min(bfs.get(a, 30), 30), min(bfs1.get(a, 30), 30), e.status == "crowded", e.status == "covered",
                  b in S.lm, tk == 1, tk == 2, tk == 3, b in set(T), e.oneway_to is not None,
                  len([1 for _, bb, ee in S.adj[b] if S.can_move(b, bb, ee)])])
    return mv, np.array(X, float).reshape(len(mv), NG)


def fit_greedy(data, epochs=80, l2=1e-2, lr=0.05, seed=2026):
    """data: list (moves, g, c, nhãn) -> W9 (NG, len(c)). Logit đa lớp trên các bước kề (tất định)."""
    import torch
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    D = [(torch.tensor(X, dtype=torch.float32), torch.tensor(c, dtype=torch.float32), mv.index(y))
         for mv, X, c, y in data if y in mv]
    W = torch.zeros(NG, len(data[0][2]), requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(epochs):
        loss = sum(torch.nn.functional.cross_entropy((X @ (W @ c))[None] * 3, torch.tensor([y])) for X, c, y in D) / len(D)
        loss = loss + l2 * (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.set_num_threads(nt)
    return W.detach().numpy().astype(np.float64)


def greedy_gaps(S, gf, W9, moves):
    mv, X = gf
    if not mv:
        return {a: BIG for a in moves}
    sc = X @ (W9 @ cvec(S))
    b = sc.max()
    d = {a: min(b - x, BIG) for a, x in zip(mv, sc)}
    return {a: d.get(a, BIG) for a in moves}


def r9_rule_cols(S, gf, moves):
    """Mỗi bước: (thuộc nhóm tốt nhất theo luật R9, thứ hạng theo luật)."""
    mv, X = gf
    m = S.mission
    keys = {}
    for a, x in zip(mv, X):
        man, euc, crowd, lm = x[0], x[1], x[4], x[6]
        k = []
        if m.fragile:
            k.append(crowd)
        if S.night:
            k.append(lm)
        k.append(man if m.urgent else round(euc, 6))
        if not S.night:
            k.append(lm)
        if not m.fragile:
            k.append(crowd)
        keys[a] = tuple(k)
    if not keys:
        return [(0.0, 3.0, 0.0) for _ in moves]
    order = sorted(set(keys.values()))
    pick = min(keys, key=lambda a: keys[a] + (ABS_ORDER.index(a),))     # hòa: Lên > Phải > Xuống > Trái (train 95,45%)
    return [(float(keys[a] == order[0]), float(order.index(keys[a])), float(a == pick)) if a in keys else (0.0, 3.0, 0.0)
            for a in moves]


ABS_ORDER = [0, 3, 1, 2]       # UP, RIGHT, DOWN, LEFT (chiều kim đồng hồ từ Bắc)


def random_tables(n=16, seed=2026):
    """XTAB=2: n bảng giá ngẫu nhiên (seed cố định) làm "cố vấn" thêm cho mô hình chọn bước."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = np.zeros(NF)
        w[0] = 1
        w[1], w[2], w[5] = rng.uniform(0, 5), rng.uniform(-0.5, 1), rng.uniform(0, 5)
        w[6], w[7], w[8] = rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(0, 6)
        w[9], w[10], w[11] = rng.uniform(0, 2), rng.uniform(0, 2), rng.uniform(0, 4)
        out.append(w)
    return out


def path_comp(M, w, moves):
    """Đặc trưng (NF cột) của lộ trình rẻ nhất theo w bắt đầu bằng từng bước; -1 nếu bước đó không có lộ trình."""
    out = np.full((len(moves), NF), -1.0)
    if M is None:
        return out
    c = M[:, :NF] @ w
    for k, a in enumerate(moves):
        sel = np.where(M[:, NF].astype(int) == a)[0]
        if len(sel):
            out[k] = M[sel[np.argmin(c[sel])], :NF]
    return out


def cond_features(S, paths, Ws, moves):
    """Đặc trưng thêm cho mỗi bước: với mỗi robot r có bảng giá Ws[r] (r = 9: điểm tham lam):
    (chênh chi phí, có phải rẻ nhất) + đêm + loại nơi giao."""
    c = cvec(S)
    cols, ext = [], {}
    for r in sorted(Ws):
        if r == 9:
            g = greedy_gaps(S, paths[2], Ws[9], moves)
        else:                            # r = 1..8: bảng giá cộng; r = 11..18: mạng nhỏ của robot r - 10
            w = weights(Ws[r], c)
            M = paths[1] if r % 10 == 4 else paths[0]
            if DIJ:
                M = merge_paths(M, dij_paths(S, w, r % 10 == 4))
            g = gaps(M, w, moves)
            if EXT and r < 9:
                ext[r] = path_comp(M, w, moves)
        cols.append([(g[a], float(g[a] < 1e-6)) for a in moves])
    if XTAB:
        for w in (XTABLES if XTAB == 1 else random_tables()):
            g = gaps(dij_paths(S, w), w, moves)
            cols.append([(g[a], float(g[a] < 1e-6)) for a in moves])
    if R9RULE:
        cols.append([c[:2] if R9RULE == 1 else c for c in r9_rule_cols(S, paths[2], moves)])
    scene = [float(S.night)] + [float(S.mission.goal == x) for x in GOALS]
    out = np.array([[v for col in cols for v in col[k]] + scene for k in range(len(moves))], float).reshape(len(moves), -1)
    if EXT:
        w0 = np.zeros(NF)
        w0[0], w0[1], w0[5] = 1, 1e-3, 1e-3          # ít đoạn nhất (phá hòa nhẹ: ít đông, ít địa điểm)
        out = np.concatenate([out] + [ext[r] for r in range(1, 9)] + [path_comp(paths[0], w0, moves)], 1)
    return out
