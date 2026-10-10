"""H25 (10/10): đọc ảnh gốc + xoay 90/180/270 độ (KHÔNG lật), đổi kết quả về khung gốc rồi biểu quyết đa số.
Một lần xoay 90° NGƯỢC chiều kim đồng hồ (PIL rotate(90, expand=True)): ô gốc (r, c) -> (C-1-c, r), lưới (R, C) -> (C, R);
hướng gốc Lên -> Trái, Phải -> Lên, Xuống -> Phải, Trái -> Xuống. Hòa -> theo ảnh gốc. Lần đọc có lưới khác đa số bị bỏ."""
from collections import Counter

BACK = {"LEFT": "UP", "UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT"}   # hướng trong ảnh đã xoay 1 lần -> hướng gốc


def _unrot1(v):
    """Kết quả đọc ảnh đã xoay 90° (ngược kim đồng hồ) -> khung trước khi xoay."""
    R2, C2 = v["grid"]                       # lưới trong ảnh xoay = (C, R) gốc
    f = lambda rc: (rc[1], R2 - 1 - rc[0]) if rc is not None else None  # noqa: E731
    return {**v, "grid": (C2, R2), "nodes": [f(n) for n in v["nodes"]],
            "edges": [{**e, "a": f(e["a"]), "b": f(e["b"]), "oneway_to": f(e["oneway_to"])} for e in v["edges"]],
            "landmarks": {k: [f(x) for x in xs] for k, xs in v["landmarks"].items()},
            "robot": f(v["robot"]), "heading": BACK[v["heading"]]}


def unrotate(v, k):
    for _ in range(k):
        v = _unrot1(v)
    return v


def _vote(vals, first):
    c = Counter(vals).most_common()
    best = [x for x, n in c if n == c[0][1]]
    return first if first in best else best[0]


def vote(reads):
    """reads[0] = ảnh gốc. -> một kết quả đọc."""
    base = reads[0]
    ok = [r for r in reads if r is not None]
    if base is None or len(ok) < 2:
        return base
    g = _vote([r["grid"] for r in ok], base["grid"])
    rs = [r for r in ok if r["grid"] == g]
    if base not in rs:
        return base                          # ảnh gốc lệch lưới đa số: giữ nguyên ảnh gốc cho an toàn
    n = len(rs)
    allnodes = sorted({x for r in rs for x in r["nodes"]})
    nodes = [x for x in allnodes if (sum(x in set(r["nodes"]) for r in rs) * 2 > n) or (x in base["nodes"] and sum(x in set(r["nodes"]) for r in rs) * 2 == n)]
    ns = set(nodes)
    E = [{frozenset((e["a"], e["b"])): e for e in r["edges"]} for r in rs]
    keys = sorted({k for d in E for k in d}, key=lambda k: sorted(k))
    edges = []
    for k in keys:
        a, b = sorted(k)
        if a not in ns or b not in ns:
            continue
        has = [k in d for d in E]
        if not _vote(has, has[0]):
            continue
        es = [d[k] for d in E if k in d]
        e0 = E[0].get(k, es[0])
        edges.append({**e0, "a": a, "b": b, "status": _vote([e["status"] for e in es], e0["status"]),
                      "stairs": _vote([e["stairs"] for e in es], e0["stairs"]),
                      "oneway_to": _vote([e["oneway_to"] for e in es], e0["oneway_to"])})
    lab = []
    for r in rs:
        m = {x: t for t, xs in r["landmarks"].items() for x in xs}
        lab.append(m)
    landmarks = {}
    for x in nodes:
        t = _vote([m.get(x) for m in lab], lab[0].get(x))
        if t is not None:
            landmarks.setdefault(t, []).append(x)
    robot = _vote([r["robot"] for r in rs], base["robot"])
    heading = _vote([r["heading"] for r in rs if r["robot"] == robot], base["heading"] if base["robot"] == robot else None)
    return {**base, "grid": g, "nodes": nodes, "edges": edges, "landmarks": landmarks, "robot": robot, "heading": heading}
