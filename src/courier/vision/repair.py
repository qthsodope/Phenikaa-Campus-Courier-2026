"""H27 (10/10): sửa bản đồ đọc từ ảnh bằng LUẬT CỦA ĐỀ + câu yêu cầu. Công tắc COURIER_REPAIR = tập con "a,b,c,d":
 a) câu nhắc loại X (nơi giao / điểm ghé) mà bản đồ không có X -> giao lộ chưa có địa điểm có P(X) cao nhất được gán X;
 b) câu có tham chiếu cho X -> X phải có đúng 2 bản: thiếu thì thêm giao lộ P(X) cao nhất (thừa đã xử lý ở ép <= 2 bản);
 c) không có đường hợp lệ robot -> (ghé) -> giao: đổi các đoạn khối ảnh ít chắc nhất (đóng / thiếu / một chiều / bậc thang),
    tối đa 2 đoạn, tới khi có đường;
 d) số địa điểm (tính cả bản) phải 8-14: thiếu thì thêm giao lộ chắc nhất (loại còn < 2 bản), thừa thì bỏ bản yếu nhất.
(Đúng 1 robot và 3 mẫu chú giải khác nhau: bộ đọc đã bảo đảm.)"""
import os
from collections import deque
from itertools import combinations

from . import common as C

ON = set(x for x in os.environ.get("COURIER_REPAIR", "").split(",") if x)


def _lm_of(v):
    return {x: t for t, xs in v["landmarks"].items() for x in xs}


def _add_type(v, t, n_target):
    """Thêm bản loại t vào giao lộ trống có P(t) cao nhất tới khi đủ n_target bản."""
    if t not in C.PLACE_TYPES or "node_p" not in v:
        return
    j = C.PLACE_TYPES.index(t) + 1
    while len(v["landmarks"].get(t, [])) < n_target:
        used = _lm_of(v)
        cand = [(p[j], rc) for rc, p in v["node_p"].items() if rc not in used and rc != v["robot"]]
        if not cand:
            return
        _, rc = max(cand)
        v["landmarks"].setdefault(t, []).append(rc)


def _path_ok(v, goals, vias):
    E = {}
    for e in v["edges"]:
        if e["status"] == "closed" or e["stairs"]:
            continue
        for a, b in ((e["a"], e["b"]), (e["b"], e["a"])):
            if e["oneway_to"] is None or e["oneway_to"] == b:
                E.setdefault(a, []).append(b)
    goals, vias = set(goals), set(vias or [])
    st = (v["robot"], 1 if not vias or v["robot"] in vias else 0)
    seen, q = {st}, deque([st])
    while q:
        n, ph = q.popleft()
        if ph == 1 and n in goals:
            return True
        for b in E.get(n, []):
            k = (b, 1 if ph == 1 or b in vias else 0)
            if k not in seen:
                seen.add(k)
                q.append(k)
    return False


def _flips(v):
    """Các cách đổi một đoạn (độ chắc của trạng thái hiện tại, hàm áp dụng)."""
    out = []
    cur = {frozenset((e["a"], e["b"])): e for e in v["edges"]}
    l2s = v.get("look2status", {})
    for pp in v.get("pairs_p", []):
        k = frozenset((pp["a"], pp["b"])); e = cur.get(k); pl = pp["pl"]
        alt = max(range(1, 4), key=lambda i: pl[i])                     # kiểu nhìn (thường/đông/mái) tốt nhất
        st_alt = l2s.get(C.LOOKS[alt - 1], "normal")
        if e is None:                                                    # đoạn bị coi là không có -> thêm
            out.append((pl[0], lambda v, pp=pp, st=st_alt: v["edges"].append(
                {"a": pp["a"], "b": pp["b"], "status": st, "stairs": False, "oneway_to": None})))
        else:
            if e["status"] == "closed":
                out.append((pl[4], lambda v, e=e, st=st_alt: e.update(status=st)))
            if e["oneway_to"] is not None:
                out.append((max(pp["po"][1:]), lambda v, e=e: e.update(oneway_to=None)))
            if e["stairs"]:
                out.append((pp["ps"], lambda v, e=e: e.update(stairs=False)))
    return sorted(out, key=lambda x: x[0])[:12]


def repair(v, p):
    if not ON or v is None or "node_p" not in v:
        return v
    import copy
    v = copy.deepcopy(v)
    goal, via = p.get("goal"), p.get("via")
    if "a" in ON:
        for t in (goal, via):
            if t and not v["landmarks"].get(t):
                _add_type(v, t, 1)
    if "b" in ON:
        for t, r in ((goal, p.get("goal_ref")), (via, p.get("via_ref"))):
            if t and r and r.get("kind") in ("north", "south", "east", "west", "near", "far"):
                _add_type(v, t, 2)
    if "d" in ON:
        n = sum(len(x) for x in v["landmarks"].values())
        while n < 8:
            used = _lm_of(v)
            cand = [(max(pr[1:]), rc, C.PLACE_TYPES[max(range(1, len(pr)), key=lambda i: pr[i]) - 1]) for rc, pr in v["node_p"].items()
                    if rc not in used and rc != v["robot"]]
            cand = [c for c in cand if len(v["landmarks"].get(c[2], [])) < 2]
            if not cand:
                break
            _, rc, t = max(cand)
            v["landmarks"].setdefault(t, []).append(rc); n += 1
        while n > 14:
            keep = {goal, via}
            cand = [(v["node_p"][rc][C.PLACE_TYPES.index(t) + 1], rc, t) for t, xs in v["landmarks"].items() for rc in xs
                    if t not in keep and rc in v["node_p"]]
            if not cand:
                break
            _, rc, t = min(cand)
            v["landmarks"][t].remove(rc); n -= 1
    if "c" in ON and goal and v["landmarks"].get(goal):
        goals, vias = v["landmarks"][goal], v["landmarks"].get(via) if via else None
        if not _path_ok(v, goals, vias):
            fl = _flips(v)
            for k in (1, 2):
                done = False
                for combo in combinations(range(len(fl)), k):
                    w = copy.deepcopy(v)
                    # áp dụng trên bản sao: tìm lại đoạn tương ứng theo (a, b)
                    for i in combo:
                        fl2 = _flips(w)
                        key = fl[i][0]
                        m = [f for f in fl2 if abs(f[0] - key) < 1e-12]
                        if m:
                            m[0][1](w)
                    if _path_ok(w, w["landmarks"][goal], w["landmarks"].get(via) if via else None):
                        v, done = w, True
                        break
                if done:
                    break
    return v
