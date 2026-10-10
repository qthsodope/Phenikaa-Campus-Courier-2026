"""Đọc câu yêu cầu tiếng Việt -> goal, via, goal_ref, via_ref, urgent, fragile.

Các bước:
  1. chuẩn hóa (không dấu, chữ thường), tách từ, sửa lỗi gõ;
  2. che tên món hàng, tìm tên địa điểm;
  3. chia câu thành mệnh đề, gán vai trò cho từng mệnh đề: gây nhiễu / điểm ghé / nơi giao;
  4. trong mệnh đề: "gần/xa X (hơn)" -> X là địa điểm MỐC; "phía bắc/trên/..." -> tham chiếu hướng;
  5. gấp / dễ vỡ: tìm cụm phủ định trước, rồi cụm khẳng định.
"""
import json
import re
from collections import Counter

from . import lexicon as L
from .normalize import normalize
from .text import tokenize, correct_tokens, find_places, find_places_fallback, match_phrase, ITEM_HINTS_PATH


def _phrases(xs):
    out = sorted({tuple(normalize(x).split()) for x in xs}, key=lambda p: (-len(p), p))
    return [p for p in out if p]


DISTRACT = _phrases(L.DISTRACT)
VIA = _phrases(L.VIA)
VIA_TAKE = _phrases(L.VIA_TAKE)
GOAL_CLAUSE = _phrases(L.GOAL_CLAUSE)
URG_NEG, URG_POS = _phrases(L.URGENT_NEG), _phrases(L.URGENT_POS)
FRA_NEG, FRA_POS = _phrases(L.FRAGILE_NEG), _phrases(L.FRAGILE_POS)
DIRS = sorted(((p, k) for k, xs in L.DIRECTIONS.items() for p in _phrases(xs)), key=lambda x: (-len(x[0]), x))
NEAR, FAR = _phrases(L.NEAR), _phrases(L.FAR)
ITEMS = _phrases(L.ITEMS)

SENT_END = {".", "!", "?", ";"}
DELIVER_VERBS = {"mang", "gui", "dua", "giao", "chuyen", "di", "toi", "den", "dem", "van", "mo", "chay",
                 "qua", "sang", "ve", "lai", "tiep", "moi"}


def _find(tokens, phrase, start=0, end=None, fuzzy=True):
    """Vị trí đầu tiên của cụm phrase trong tokens[start:end], -1 nếu không có.
    fuzzy: cho phép 1 từ bị gõ sai (mất 1 chữ / đảo 2 chữ kề) nếu cụm đủ dài."""
    end = len(tokens) if end is None else end
    n = len(phrase)
    for i in range(start, end - n + 1):
        if tuple(tokens[i:i + n]) == phrase or (fuzzy and match_phrase(tokens, i, phrase, allow_real=n >= 3)):
            return i
    return -1


def _has_any(tokens, phrases):
    return any(_find(tokens, p) >= 0 for p in phrases)


def _clauses(tokens):
    """Chia thành mệnh đề: theo dấu câu, dấu phẩy, 'sau đó', 'rồi + động từ giao'."""
    out, cur = [], []
    for i, w in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if w in SENT_END or w == ",":
            if cur:
                out.append(cur)
            cur = []
            continue
        prv = tokens[i - 1] if i > 0 else ""
        if cur and ((w == "sau" and nxt == "do") or (w == "roi" and prv != "da" and (nxt in DELIVER_VERBS or prv == "truoc"))
                    or (w == "nhung" and nxt == "phai")):
            out.append(cur)
            cur = []
        cur.append(i)
        if w == "thi" and i > 0 and tokens[i - 1] == "xong":    # "lấy ... xong thì | đưa ..."
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _flag(tokens, neg, pos):
    """Có cụm phủ định -> False; có cụm khẳng định -> True; không nhắc -> False."""
    t = list(tokens)
    found_neg = False
    for p in neg:
        while True:
            i = _find(t, p)
            if i < 0:
                break
            found_neg = True
            t[i:i + len(p)] = ["_"] * len(p)
    if found_neg:
        return False
    return _has_any(t, pos)


NEGATORS = {"khong", "chang", "cha", "chua", "dung", "khoi", "ko", "kg", "kh", "k"}
_DEADLINE_RE = [re.compile(r"\btruoc \d+ ?(h|gio)\b"), re.compile(r"\btruoc \d+h\d*\b"),
                re.compile(r"\btrong (vong )?\d+ phut\b")]


def flag_tri(tokens, neg, pos, deadline=False):
    """Bước 5C: đọc một cờ (gấp / dễ vỡ) -> True / False / None (không thấy cụm nào).
    - Khớp các cụm khẳng định và phủ định, cụm dài trước, không chồng lấn.
    - PHỦ ĐỊNH CÓ PHẠM VI: đếm từ phủ định (không, chẳng, chưa, đừng...) trong 3 từ ngay trước cụm, cùng mệnh đề;
      mỗi từ phủ định lật nghĩa một lần, nên phủ định kép "không phải là không gấp" = gấp.
    - deadline: hạn giờ dạng số ("trước 9h", "trong 10 phút") = gấp.
    Có ít nhất một ý phủ định -> False; có ý khẳng định -> True; không thấy gì -> None."""
    phrases = sorted([(p, False) for p in neg] + [(p, True) for p in pos], key=lambda x: (-len(x[0]), x))
    used = [False] * len(tokens)
    hits = []
    for p, pol in phrases:
        start = 0
        while True:
            i = _find(tokens, p, start)
            if i < 0:
                break
            start = i + 1
            if any(used[i:i + len(p)]):
                continue
            for k in range(i, i + len(p)):
                used[k] = True
            hits.append((i, pol))
    if deadline:
        joined = " ".join(tokens)
        for rx in _DEADLINE_RE:
            for m in rx.finditer(joined):
                hits.append((len(joined[:m.start()].split()), True))
    res = []
    for i, pol in hits:
        nneg = 0
        for k in range(i - 1, max(-1, i - 4), -1):
            w = tokens[k]
            if w in SENT_END or w in (",", ":"):
                break
            if not used[k] and w in NEGATORS:
                nneg += 1
        res.append(pol if nneg % 2 == 0 else not pol)
    if not res:
        return None
    return False if not all(res) else True


def _item_mask(tokens):
    mask = [False] * len(tokens)
    for p in ITEMS:
        i = _find(tokens, p)
        while i >= 0:
            for k in range(i, i + len(p)):
                mask[k] = True
            i = _find(tokens, p, i + 1)
    return mask


_ITEM_HINTS = None
# Lưới an toàn: đoán loại địa điểm cho cụm từ lạ (xem safety.py). Mặc định bật phương án "char",
# dựng lần đầu khi cần (học từ train + val-A, vài giây). "default" = chưa dựng.
_NET = "default"


def set_net(kind, **kw):
    """Đổi lưới an toàn: None (tắt) hoặc tên phương án trong safety.py ("char", "e5", "ens")."""
    global _NET
    if kind is None:
        _NET = None
        return
    from .safety import build_net
    _NET = build_net(kind, **kw)


def _get_net():
    global _NET
    if _NET == "default":
        from .safety import build_default_net
        from . import aux
        _NET = aux.get("safety") or build_default_net()
    return _NET



def _goal_from_item(tk, mentions):
    """Không thấy tên nơi giao nào: tìm món hàng quen thuộc (vd 'khay cơm' -> căn tin)."""
    global _ITEM_HINTS
    if _ITEM_HINTS is None:
        raw = json.loads(ITEM_HINTS_PATH.read_text(encoding="utf-8"))
        _ITEM_HINTS = sorted(((tuple(k.split()), v) for k, v in raw.items()), key=lambda x: (-len(x[0]), x))
    for phrase, typ in _ITEM_HINTS:
        if _find(tk, phrase) >= 0:
            return typ
    return None


def parse_tokens(text):
    return correct_tokens(tokenize(normalize(text)))


def parse(text):
    """Trả về dict giống scenes.json[mission] (không có rc trong ref):
    {goal, via, goal_ref: {kind, anchor}|None, via_ref, urgent, fragile}."""
    tk = parse_tokens(text)
    mask = _item_mask(tk)
    places = find_places(tk, mask)
    used = list(mask)
    for a, b, _ in places:
        for k in range(a, b):
            used[k] = True
    places = sorted(places + find_places_fallback(tk, used))
    net = _get_net()
    if net is not None:                       # lưới an toàn: đoán loại cho cụm lạ ở "ô trống"
        from .safety import slot_spans
        for a, b, _ in places:
            for k in range(a, b):
                used[k] = True
        spans = [sp for sp in slot_spans(tk, used)]
        for (a, b), t in zip(spans, net.predict([" ".join(tk[a:b]) for a, b in spans])):
            if t is not None and not any(used[a:b]):
                places.append((a, b, t))
                for k in range(a, b):
                    used[k] = True
        places.sort()
    place_at = {a: (a, b, t) for a, b, t in places}

    mentions = []   # dict: start, end, type, role, ref
    for clause in _clauses(tk):
        ctoks = [tk[i] for i in clause]
        cset = set(clause)
        if _has_any(ctoks, DISTRACT):
            role = "distract"
        elif _has_any(ctoks, GOAL_CLAUSE):
            role = "goal"
        elif _has_any(ctoks, VIA) or (_has_any(ctoks, VIA_TAKE) and ("truoc" in ctoks or "xong" in ctoks)
                                       and _find(ctoks, ("truoc", "khi")) < 0):
            role = "via"
        else:
            role = "goal"
        cms = [place_at[i] for i in clause if i in place_at]
        for a, b, t in cms:
            mentions.append({"start": a, "end": b, "type": t, "role": role, "ref": None,
                             "clause_end": clause[-1], "cset": cset})

    # mốc (gần/xa X) và hướng (phía bắc ...)
    for k, m in enumerate(mentions):
        a = m["start"]
        before = tk[max(0, a - 3):a]
        kind = None
        for p in sorted(NEAR + FAR, key=lambda p: (-len(p), p)):
            n = len(p)
            if n <= len(before) and tuple(before[-n:]) == p:
                kind = "near" if p in NEAR else "far"
                break
            if n + 1 <= len(before) and tuple(before[-n - 1:-1]) == p and before[-1] in ("voi", "hon"):
                kind = "near" if p in NEAR else "far"
                break
        if kind:
            m["role"] = "anchor"
            # gắn vào địa điểm (không phải mốc) đứng ngay trước trong cùng mệnh đề
            for j in range(k - 1, -1, -1):
                prev = mentions[j]
                if prev["role"] != "anchor" and a in prev["cset"]:
                    prev["ref"] = {"kind": kind, "anchor": m["type"]}
                    break
    for k, m in enumerate(mentions):
        if m["role"] == "anchor" or m["ref"]:
            continue
        nxt_start = mentions[k + 1]["start"] if k + 1 < len(mentions) else len(tk)
        lo, hi = m["end"], min(nxt_start, m["clause_end"] + 1, m["end"] + 6)
        for p, kind in DIRS:
            if _find(tk, p, lo, hi) >= 0:
                m["ref"] = {"kind": kind, "anchor": None}
                break

    def pick(role):
        ms = [m for m in mentions if m["role"] == role]
        if not ms:
            return None, None
        cnt = Counter(m["type"] for m in ms)
        best = max(cnt.values())
        t = next(m["type"] for m in ms if cnt[m["type"]] == best)
        ref = next((m["ref"] for m in ms if m["type"] == t and m["ref"]), None)
        return t, ref

    goal, goal_ref = pick("goal")
    via, via_ref = pick("via")
    cands = _candidates(mentions, tk)
    distract = sorted({m["type"] for m in mentions if m["role"] == "distract"})
    if goal is None and via is not None:      # chỉ thấy 1 địa điểm -> coi là nơi giao
        goal, goal_ref, via, via_ref = via, via_ref, None, None
    if via == goal:
        via, via_ref = None, None
    if goal is None:                          # dự phòng: đoán nơi giao theo món hàng (học từ train)
        goal = _goal_from_item(tk, mentions)

    out = {
        "goal": goal, "via": via, "goal_ref": goal_ref, "via_ref": via_ref,
        "urgent": _flag(tk, URG_NEG, URG_POS),
        "fragile": _flag(tk, FRA_NEG, FRA_POS),
        "_cands": cands,
        "distract": distract,
    }
    from .mode import FLAG_V2
    if FLAG_V2:
        tri = {"urgent": flag_tri(tk, URG_NEG, URG_POS, deadline=True), "fragile": flag_tri(tk, FRA_NEG, FRA_POS)}
        out["urgent"], out["fragile"] = bool(tri["urgent"]), bool(tri["fragile"])
        out["_rule_flags"] = tri
        from .flagnet import get_flagnet
        fn = get_flagnet()
        if fn is not None:                    # mô hình n-gram ký tự nhỏ đọc từng câu nhỏ (nguồn thứ ba)
            out["_char_flags"] = fn.predict_mission(text)
    from .model import get_predictor
    pred = get_predictor()
    if pred is not None:                      # mô hình ngôn ngữ (bước 5B): xác suất cho giải mã chung với bản đồ
        out["_probs"] = pred.probs([text])[0]
    return out


def _candidates(mentions, tk):
    """Các phương án xếp hạng cho kiểm tra chéo với bản đồ (resolve.apply_constraints):
    goal: loại ở mệnh đề nơi giao (nhắc nhiều trước), rồi điểm ghé, rồi đoán theo món hàng;
    via: loại ở mệnh đề điểm ghé; ref: tham chiếu đã đọc cho từng loại."""
    def ranked(role):
        ms = [m for m in mentions if m["role"] == role]
        cnt = Counter(m["type"] for m in ms)
        first = {}
        for k, m in enumerate(ms):
            first.setdefault(m["type"], k)
        return sorted(cnt, key=lambda t: (-cnt[t], first[t]))
    g, v = ranked("goal"), ranked("via")
    goal = g + [t for t in v if t not in g]
    item = _goal_from_item(tk, mentions)
    if item and item not in goal:
        goal.append(item)
    refs = {}
    for m in mentions:
        if m["ref"] and m["role"] in ("goal", "via"):
            refs.setdefault((m["role"], m["type"]), m["ref"])
    return {"goal": goal, "via": v, "refs": refs}
