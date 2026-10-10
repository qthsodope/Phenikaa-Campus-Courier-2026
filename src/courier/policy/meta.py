"""Mô hình "chọn bước" cho V2 (robot có luật chưa giải mã hết thành lời).

Ý tưởng: robot nào cũng "tìm đường rẻ nhất theo một bảng giá nào đó", nhưng bảng giá và cách phá hòa đổi theo điều
kiện. Ta tính sẵn MỘT DÀN bộ tìm đường với nhiều bảng giá (BANK), mỗi bộ cho biết bước đi nào rẻ nhất và mỗi bước đắt
hơn bao nhiêu. Cùng với đặc trưng của bước đi (rẽ / thẳng / quay đầu, loại đoạn, có đi vào địa điểm…) và của cảnh (mưa,
gấp, dễ vỡ, có điểm ghé, hướng mũi), một cây tăng cường nhỏ cho từng robot học xem robot đó đi theo bộ nào, khi nào.

Chỉ học từ nhãn train (và validation khi làm bản nộp). Tất định (random_state cố định).
"""
import math

import numpy as np

from .plan import first_move_costs, weighted, prepare
from .solver import turn_kind

BIG = 50.0

# dàn bộ tìm đường: (tên, bảng giá, đi được bậc thang?)
BANK = [
    ("deu", weighted(), False),
    ("deu_ddiem3", weighted(landmark=3), False),
    ("dong2", weighted(crowded=2), False),
    ("dong4", weighted(crowded=4), False),
    ("dong7", weighted(crowded=7), False),
    ("mai05", weighted(covered=0.5), False),
    ("mai02", weighted(covered=0.2), False),
    ("mai05_dong2", weighted(covered=0.5, crowded=2), False),
    ("quay3", weighted(back=3), False),
    ("quay3_re05", weighted(back=3, right=0.5, left=0.5), False),
    ("re3_quay30", weighted(right=3, left=3, back=30), False),
    ("phai05_trai45", weighted(right=0.5, left=4.5, back=9), False),
    ("trai05_phai45", weighted(right=4.5, left=0.5, back=9), False),
    ("r1fit", weighted(crowded=6, landmark=3, right=0.8, left=0.2, back=3.4), False),
    ("r2fit", weighted(crowded=1.8, covered=0.3, landmark=2.6, right=0.5, left=0.2, back=0.8), False),
    ("r3fit", weighted(crowded=3.7, landmark=3.6, right=0.15, left=0.7, back=2.2), False),
    ("dong3_ddiem3_quay3", weighted(crowded=3, landmark=3, back=3), False),
    ("chan_deu", weighted(), True),
    ("chan_dong2", weighted(crowded=2), True),
    ("chan_ddiem3", weighted(landmark=3), True),
    ("chan_bac3", weighted(stairs=3), True),
    ("chan_bac5_dong2", weighted(stairs=5, crowded=2), True),
]
def v1_like(rid, landmark=3.0):
    """Bộ tìm đường "bảng giá V1 của robot rid (đổi theo mưa / gấp / dễ vỡ như V1) + phí đi xuyên địa điểm".
    Dò trên train V2: khớp tốt với R4 (khi mưa bậc thang +4), R5 (rẽ +3, quay +30), R7 dễ vỡ, R8 (rẽ trái +4,5)."""
    from .robots import robot_config

    def f(S, u, v, e, h, a, ph):
        edge, turn, legged = robot_config(rid, S)
        return edge(e) + (turn(h, a) if turn else 0.0) + (landmark if v in S.lm else 0.0)
    return f


BANK += [(f"v1r{r}_dd3", v1_like(r), r == 4) for r in range(1, 9)]
# Mô hình TƯỜNG MINH tốt nhất tìm được cho từng robot (dò trên train V2, chấm độ chính xác thật có phá hòa):
# (điều kiện -> (đông, mái, phí đi xuyên địa điểm, phí rẽ (phải, trái, quay đầu), bậc thang thêm, đi bậc?, phá hòa)).
# Bậc phụ sau giá chính: giống robot 0. Dùng làm ĐẶC TRƯNG cho mô hình chọn bước.
EXPLICIT = {
    1: {None: (3, 1, 3, (0, 0, 2), 0, False, ("mui", (0, 1, 2, 3)))},
    2: {None: (2, 1, 3, (0, 0, 0), 0, False, ("td", (1, 2, 3, 0)))},
    3: {True: (2, 0.3, 0, (0, 0, 0), 0, False, ("mui", (1, 0, 2, 3))), False: (2, 1, 3, (0, 0, 0), 0, False, ("mui", (2, 0, 1, 3)))},
    4: {True: (2, 1, 3, (0, 0, 0), 4, True, ("td", (2, 3, 1, 0))), False: (1, 1, 3, (0, 0, 0), 0, True, ("mui", (0, 1, 2, 3)))},
    5: {None: (5, 1, 3, (3, 3, 30), 0, False, ("mui", (0, 2, 1, 3)))},
    6: {True: (2, 1, 0, (0, 0, 5), 0, False, ("mui", (0, 1, 2, 3))), False: (5, 1, 3, (1.5, 1.5, 15), 0, False, ("mui", (1, 2, 0, 3)))},
    7: {True: (7, 1, 3, (1.5, 1.5, 15), 0, False, ("mui", (0, 2, 1, 3))), False: (2, 1, 0, (0, 0, 0), 0, False, ("mui", (0, 1, 2, 3)))},
    8: {None: (3, 1, 3, (0.5, 4.5, 9), 0, False, ("mui", (1, 2, 3, 0)))},
}
EXPLICIT_COND = {3: lambda S: S.rain, 4: lambda S: S.rain, 6: lambda S: S.mission.urgent, 7: lambda S: S.mission.fragile}


def explicit_pick(rid, S):
    cfg = EXPLICIT[rid]
    C, V, P, TT, ST, legged, (mode, perm) = cfg[EXPLICIT_COND[rid](S) if rid in EXPLICIT_COND else None]

    def step(S_, u, v, e, h, a, p):
        tk = turn_kind(h, a)
        prim = {"normal": 1, "crowded": C, "covered": V}[e.status] + (P if v in S_.lm else 0) \
            + (0, TT[0], TT[1], TT[2])[tk] + (ST if e.stairs else 0)
        cr = int(e.status == "crowded")
        return (round(prim, 6), int(v in S_.lm), (-int(e.status == "covered") if S_.rain else cr), cr, int(tk == 3))
    c = first_move_costs(S, S.vias, S.goals, step, legged) if S.goals else {}
    if not c:
        return None
    m = min(c.values())
    o = [a for a, x in c.items() if x == m]
    return min(o, key=(lambda a: perm.index(turn_kind(S.heading, a))) if mode == "mui" else (lambda a: perm.index(a)))


SCENE_KEYS = ["rain", "urgent", "fragile", "has_via", "heading", "n_moves", "n_goal", "n_via", "goal_ref", "via_ref"]


def _targets_now(S):
    return S.vias if S.vias else S.goals


def scene_rows(S, r0_pick=None):
    """-> (danh sách bước đi hợp lệ, ma trận đặc trưng mỗi bước). S phải đã prepare()."""
    moves = [a for a, b, e in S.adj[S.robot] if S.can_move(S.robot, b, e) or S.can_move(S.robot, b, e, True)]
    nb = {a: (b, e) for a, b, e in S.adj[S.robot]}
    T = _targets_now(S) or [S.robot]
    m = S.mission
    scene = [S.rain, m.urgent, m.fragile, m.via is not None, S.heading, len(moves), len(S.goals),
             len(S.vias or []), m.goal_rc is not None, m.via_rc is not None]
    bank = []
    for name, fn, legged in BANK:
        c = first_move_costs(S, S.vias, S.goals, fn, legged) if S.goals else {}
        best = min(c.values()) if c else 0.0
        bank.append({a: (min(c[a] - best, BIG) if a in c else BIG) for a in moves})
    picks = [explicit_pick(r, S) for r in range(1, 9)]
    rows = []
    d0 = min(abs(S.robot[0] - t[0]) + abs(S.robot[1] - t[1]) for t in T)
    e0 = min(math.dist(S.robot, t) for t in T)
    for a in moves:
        b, e = nb[a]
        tk = turn_kind(S.heading, a)
        f = [tk, a, ["normal", "crowded", "covered", "closed"].index(e.status), e.stairs, e.oneway_to is not None,
             S.can_move(S.robot, b, e), b in S.lm, b in set(S.goals) | set(S.vias or []),
             min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in T) - d0, min(math.dist(b, t) for t in T) - e0,
             len([1 for _, bb, ee in S.adj[b] if S.can_move(b, bb, ee)]),
             int(r0_pick == a) if r0_pick is not None else -1] + [int(ep == a) for ep in picks]
        for bk in bank:
            f += [bk[a], float(bk[a] < 1e-9)]
        rows.append(f + scene)
    return moves, np.array(rows, dtype=float)


FEATURE_NAMES = (["turn", "abs", "status", "stairs", "oneway", "passable", "next_lm", "next_target", "d_manh", "d_eucl",
                  "deg_next", "r0_pick"] + [f"exp{r}_pick" for r in range(1, 9)] + [f"{n}_{k}" for n, _, _ in BANK for k in ("d", "opt")] + SCENE_KEYS)


class MetaPolicy:
    """Mỗi robot một cây tăng cường: điểm cho từng bước, chọn bước điểm cao nhất."""

    def __init__(self, robots=range(1, 10), seed=2026, Ws=None, modes=None):
        self.robots = list(robots)
        self.seed = seed
        self.models = {}
        self.Ws = Ws or {}          # bảng giá có điều kiện (cond.py) cho robot 1-8, dùng làm đặc trưng
        # 10/10 (thử nghiệm theo bài khác): kiểu mô hình từng robot. "base" = 1 cây; "bag" = 5 cây (mỗi cây 70% đặc
        # trưng) lấy trung bình; "pair" = xếp hạng theo cặp (học "bước nhãn tốt hơn bước kia" trên hiệu đặc trưng);
        # "deep" = cây 31 lá, 150 vòng; "big" = 200 vòng. Tiền tố "x" = dùng thêm đặc trưng v2 (cond.EXT).
        self.modes = dict(modes or {})

    def fit(self, data):
        """data: list (moves, X, labels[10])."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        def hgb(seed, **kw):
            p = dict(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40,
                     l2_regularization=1.0, random_state=seed)
            p.update(kw)
            return HistGradientBoostingClassifier(**p)
        from .cond import EXT
        self.ext = EXT
        for r in self.robots:
            mode = self.modes.get(r, "base")
            cut = self._cut(r, data[0][1].shape[1] if data else 0)
            data_r = [(mv, x[:, :cut], lab) for mv, x, lab in data]
            mode = mode[1:] if mode.startswith("x") else mode
            X = np.concatenate([x for _, x, _ in data_r])
            y = np.concatenate([[int(a == lab[r]) for a in mv] for mv, _, lab in data])
            if mode == "deep":
                self.models[r] = hgb(self.seed, max_iter=150, learning_rate=0.06, max_leaf_nodes=31,
                                     min_samples_leaf=20).fit(X, y)
            elif mode == "big":
                self.models[r] = hgb(self.seed, max_iter=200, learning_rate=0.05).fit(X, y)
            elif mode == "bag":
                self.models[r] = [hgb(s, max_features=0.7).fit(X, y) for s in range(5)]
            elif mode == "pair":
                P, Y = [], []
                for mv, x, lab in data_r:
                    if lab[r] not in mv:
                        continue
                    k = mv.index(lab[r])
                    for j in range(len(mv)):
                        if j != k:
                            P += [x[k] - x[j], x[j] - x[k]]
                            Y += [1, 0]
                self.models[r] = ("pair", hgb(self.seed, max_iter=150).fit(np.array(P), np.array(Y)))
            else:
                self.models[r] = hgb(self.seed).fit(X, y)
        return self

    def _cut(self, r, ncol):
        """Số cột robot r dùng: robot kiểu "x…" dùng cả đặc trưng v2 (cond.EXT); robot khác bỏ 108 cột cuối."""
        from .cond import EXT_COLS
        if getattr(self, "ext", 0) and not getattr(self, "modes", {}).get(r, "base").startswith("x"):
            return ncol - EXT_COLS
        return ncol

    def predict(self, moves, X, r, default):
        if not moves:
            return default
        X = X[:, :self._cut(r, X.shape[1])]
        m = self.models[r]
        if isinstance(m, tuple):          # xếp hạng theo cặp: điểm = tổng xác suất "thắng" các bước còn lại
            n = len(moves)
            p = m[1].predict_proba((X[:, None, :] - X[None, :, :]).reshape(n * n, -1))[:, 1].reshape(n, n)
            np.fill_diagonal(p, 0)
            p = p.sum(1)
        elif isinstance(m, list):
            p = np.mean([x.predict_proba(X)[:, 1] for x in m], 0)
        else:
            p = m.predict_proba(X)[:, 1]
        return moves[int(np.argmax(p))]


def build_rows(S):
    prepare(S)
    from .rules_v2 import robot0_move
    return scene_rows(S, robot0_move(S))


def with_cond(S, moves, X, paths, Ws):
    """Ghép đặc trưng của bảng giá có điều kiện (cond.py) vào ma trận đặc trưng X."""
    from .cond import cond_features
    if not moves:
        return X
    return np.concatenate([X, cond_features(S, paths, Ws, moves)], 1)
