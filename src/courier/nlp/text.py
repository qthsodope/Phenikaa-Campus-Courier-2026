"""Tách từ, sửa lỗi gõ, tìm tên địa điểm trong câu đã chuẩn hóa."""
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import lexicon as L
from .lexicon import ALIASES
from .normalize import normalize

TOKEN_RE = re.compile(r"[a-z0-9]+|[.,!?;:]")
ITEM_HINTS_PATH = Path(__file__).with_name("item_hints.json")


def tokenize(s):
    return TOKEN_RE.findall(s)


# ----------------------------------------------------------------- âm tiết tiếng Việt hợp lệ
# Phụ âm đầu + vần (đã bỏ dấu). Từ KHÔNG thể là âm tiết tiếng Việt (vd "khogn", "cag") chắc chắn là lỗi gõ.
_ONSETS = "ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x"
_RHYMES = ("a ac ach ai am an ang anh ao ap at au ay e ec em en eng eo ep et eu i ia ich iec iem ien iep iet "
           "ieu ieng im in inh ip it iu o oa oac oach oai oam oan oang oanh oao oap oat oay oc oe oem oen oeo oet oi "
           "om on ong ooc oong op ot u ua uac uan uang uanh uat uay uc ue uech uen uenh uet ui um un ung uo uoc "
           "uoi uom uon uong uop uot up ut uu uy uya uych uyen uyet uynh uyt uyu y yem yen yet yeu").split()
_SYL_RE = re.compile(rf"^(?:{_ONSETS})?(?:{'|'.join(sorted(_RHYMES, key=len, reverse=True))})$")


def is_syllable(w):
    return bool(_SYL_RE.match(w))


# ----------------------------------------------------------------- từ vựng
def build_vocab(texts, extra=(), min_count=3):
    """Kho từ vựng = từ xuất hiện >= min_count lần trong train + mọi từ trong từ điển;
    kèm số lần mỗi cặp từ liền nhau (để sửa lỗi gõ theo ngữ cảnh)."""
    uni, bi = Counter(), Counter()
    for t in texts:
        w = [x for x in tokenize(normalize(t))]
        uni.update(x for x in w if x.isalpha())
        bi.update(f"{a} {b}" for a, b in zip(w, w[1:]))
    for phrase in extra:
        w = tokenize(normalize(phrase))
        for x in w:
            if x.isalpha():
                uni[x] = max(uni[x], min_count)
        for a, b in zip(w, w[1:]):
            bi[f"{a} {b}"] = max(bi[f"{a} {b}"], 1)
    uni = {w: n for w, n in uni.items() if n >= min_count}
    bi = {k: n for k, n in bi.items() if all(x in uni or not x.isalpha() for x in k.split())}
    return {"uni": uni, "bi": bi}


_VOCAB = None


def lexicon_phrases():
    """Mọi cụm từ của từ điển (theo chế độ hiện tại) — được đưa vào kho từ vựng."""
    extra = [a for xs in L.PLACES.values() for a in xs]
    for name in ("ITEMS", "DISTRACT", "VIA", "VIA_TAKE", "GOAL_CLAUSE", "URGENT_NEG", "URGENT_POS", "PLACE_HEADS",
                 "PLACE_STOP", "FRAGILE_NEG", "FRAGILE_POS", "NEAR", "FAR"):
        extra += getattr(L, name)
    extra += [p for xs in L.DIRECTIONS.values() for p in xs]
    extra += [p for xs in L.PLACE_KEYWORDS.values() for p in xs]
    return extra


def vocab():
    """Kho từ vựng: dựng trong bộ nhớ từ câu TRAIN + từ điển của chế độ hiện tại (< 1 giây)."""
    global _VOCAB
    if _VOCAB is None:
        from . import aux
        _VOCAB = aux.get("vocab")
        if _VOCAB is None:
            from .mode import train_texts
            _VOCAB = build_vocab(train_texts(), lexicon_phrases())
    return _VOCAB


def _edits1(w):
    """Các từ gốc có thể đã bị gõ sai thành w. Lỗi gõ trong dữ liệu chỉ có 2 kiểu:
    mất 1 chữ, hoặc đảo 2 chữ kề nhau. Nên từ gốc = w chèn thêm 1 chữ, hoặc w đảo lại 2 chữ kề."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = set()
    for i in range(len(w) + 1):
        for ch in letters:
            out.add(w[:i] + ch + w[i:])
        if i < len(w) - 1:
            out.add(w[:i] + w[i + 1] + w[i] + w[i + 2:])
    out.discard(w)
    return out


def is_typo_of(t, w):
    """t có phải là w bị gõ sai đúng 1 lỗi (mất 1 chữ / đảo 2 chữ kề) không."""
    if t == w:
        return False
    if len(t) == len(w) - 1:
        return any(w[:i] + w[i + 1:] == t for i in range(len(w)))
    if len(t) == len(w):
        d = [i for i in range(len(w)) if t[i] != w[i]]
        return len(d) == 2 and d[1] == d[0] + 1 and t[d[0]] == w[d[1]] and t[d[1]] == w[d[0]]
    return False


def match_phrase(tokens, i, phrase, min_chars=5, allow_real=False):
    """tokens[i:] có khớp cụm phrase không: khớp hẳn, hoặc đúng 1 từ bị gõ sai 1 lỗi
    (chỉ khi cụm đủ dài >= min_chars chữ, và từ sai là từ LẠ không có trong kho — từ có nghĩa thì
    để bước sửa theo ngữ cảnh lo, tránh nhầm kiểu 'qua nhà' ~ 'qua nhận')."""
    n = len(phrase)
    if i + n > len(tokens):
        return False
    win = tokens[i:i + n]
    diff = [k for k in range(n) if win[k] != phrase[k]]
    if not diff:
        return True
    if len(diff) > 1 or sum(len(x) for x in phrase) < min_chars:
        return False
    k = diff[0]
    if not allow_real and win[k] in vocab()["uni"]:
        return False
    return len(phrase[k]) >= (2 if allow_real else 4) and is_typo_of(win[k], phrase[k])


@lru_cache(maxsize=200000)
def _candidates(w):
    U = vocab()["uni"]
    return tuple(sorted(c for c in _edits1(w) if c in U))


def correct_tokens(tokens):
    """Sửa lỗi gõ từng từ.
    - Từ lạ (không có trong kho) -> từ trong kho cách 1 lỗi gõ; nhiều ứng viên thì ưu tiên ứng viên
      hay đi cùng từ trước/sau, rồi tới ứng viên phổ biến.
    - Từ có trong kho nhưng "lạc ngữ cảnh" (vd 'hông cần' thay vì 'không cần'): nếu từ đó gần như
      không bao giờ đi cùng hai từ bên cạnh, còn một từ cách 1 lỗi gõ thì rất hay đi cùng -> thay."""
    U, B = vocab()["uni"], vocab()["bi"]
    out = []
    for i, w in enumerate(tokens):
        if not w.isalpha() or len(w) <= 1:
            out.append(w)
            continue
        prev = out[-1] if out else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""

        def ctx(c):
            return B.get(f"{prev} {c}", 0) + B.get(f"{c} {nxt}", 0)
        def ctx2(c):   # khớp cả hai phía: lấy phía yếu hơn
            return min(B.get(f"{prev} {c}", 0), B.get(f"{c} {nxt}", 0))
        cands = _candidates(w)
        if w in U:
            if len(w) >= 3 and cands:
                best = max(cands, key=lambda c: (ctx2(c), ctx(c), U[c], c))
                if (best in _cue_words() and ctx(best) >= max(REAL_WORD_MIN, 20 * ctx(w)) and U[best] > U[w]
                        and B.get(f"{prev} {best}", 0) > 0 and B.get(f"{best} {nxt}", 0) > 0):
                    w = best
            out.append(w)
            continue
        if not cands:
            out.append(w)
            continue
        if is_syllable(w):
            # từ tiếng Việt hợp lệ nhưng lạ (vd "hay"): chỉ sửa khi ngữ cảnh hai phía ủng hộ rõ
            best = max(cands, key=lambda c: (ctx2(c), ctx(c), U[c], c))
            out.append(best if ctx2(best) >= 1 else w)
            continue
        out.append(max(cands, key=lambda c: (ctx(c), U[c], c)))
    return out


REAL_WORD_MIN = 5
_CUES = None


def _cue_words():
    """Từ 'quan trọng' được phép sửa theo ngữ cảnh: từ trong các cụm gợi ý vai trò / gấp / dễ vỡ /
    hướng và từ đầu của tên địa điểm (phòng, khu, nơi...)."""
    global _CUES
    if _CUES is None:
        xs = (L.DISTRACT + L.VIA + L.VIA_TAKE + L.GOAL_CLAUSE + L.URGENT_NEG + L.URGENT_POS + L.FRAGILE_NEG
              + L.FRAGILE_POS + L.PLACE_HEADS + [p for v in L.DIRECTIONS.values() for p in v])
        _CUES = {w for x in xs for w in tokenize(normalize(x)) if len(w) >= 3}
    return _CUES


# ----------------------------------------------------------------- địa điểm
_ALIAS_TOKENS_FULL = [(tuple(a.split()), t) for a, t in ALIASES]
_ALIAS_TOKENS = list(_ALIAS_TOKENS_FULL)


def set_aliases(aliases):
    """Thay từ điển tên địa điểm (dùng cho thí nghiệm 'giả lập tên mới')."""
    global _ALIAS_TOKENS
    _ALIAS_TOKENS = sorted(aliases, key=lambda p: (-len(" ".join(p[0])), p))


def current_aliases():
    return list(_ALIAS_TOKENS)


def find_places(tokens, blocked=None):
    """Tìm tên địa điểm (khớp dài nhất trước, không chồng lấn).
    Lượt 1 khớp hẳn; lượt 2 cho phép 1 từ bị gõ sai (mất 1 chữ / đảo 2 chữ).
    blocked: các vị trí bỏ qua (vd đã là tên món hàng).
    Trả về danh sách (vị trí bắt đầu, vị trí kết thúc, loại)."""
    found = []
    used = list(blocked) if blocked else [False] * len(tokens)
    for exact in (True, False):
        for al, typ in _ALIAS_TOKENS:          # đã xếp dài trước
            n = len(al)
            for i in range(len(tokens) - n + 1):
                if any(used[i:i + n]):
                    continue
                ok = (tuple(tokens[i:i + n]) == al if exact
                      else match_phrase(tokens, i, al, min_chars=5, allow_real=n >= 2)
                      and (n > 1 or len(al[0]) >= 6))
                if ok:
                    found.append((i, i + n, typ))
                    for k in range(i, i + n):
                        used[k] = True
    found.sort()
    return found


# ----------------------------------------------------------------- đoán theo từ khóa
def _tok_phrases(xs):
    return sorted({tuple(normalize(x).split()) for x in xs}, key=lambda p: (-len(p), p))


_HEADS = _tok_phrases(L.PLACE_HEADS)
_KEYWORDS = sorted(((p, t) for t, xs in L.PLACE_KEYWORDS.items() for p in _tok_phrases(xs)),
                   key=lambda x: (-len(x[0]), x))
_STOP = {normalize(x) for x in L.PLACE_STOP}
MAX_SPAN = 6


def find_places_fallback(tokens, used):
    """Tên gọi lạ: '<từ đầu> + vài từ' có chứa từ khóa của một loại địa điểm.
    used: các vị trí đã thuộc tên địa điểm / món hàng (bỏ qua). Trả về (đầu, cuối, loại)."""
    out = []
    used = list(used)
    i = 0
    while i < len(tokens):
        head = next((h for h in _HEADS if tuple(tokens[i:i + len(h)]) == h
                     and not any(used[i:i + len(h)])), None)
        if head is None:
            i += 1
            continue
        j = i + len(head)
        hit = None
        while j < len(tokens) and j < i + len(head) + MAX_SPAN and not used[j] and tokens[j].isalnum():
            kw = next(((p, t) for p, t in _KEYWORDS if tuple(tokens[j:j + len(p)]) == p
                       and not any(used[j:j + len(p)])), None)
            if kw:
                hit = (i, j + len(kw[0]), kw[1])
                break
            if tokens[j] in _STOP:
                break
            j += 1
        if hit:
            out.append(hit)
            for k in range(hit[0], hit[1]):
                used[k] = True
            i = hit[1]
        else:
            i += 1
    return out
