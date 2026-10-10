"""Sinh câu luyện tập cho mô hình ngôn ngữ của khối 2 (bước 5B).

Hai nguồn:
  1. Câu THẬT (train; chế độ cuối thêm validation), biến đổi giữ nguyên nhãn: thay tên địa điểm bằng tên khác cùng
     loại, thay cụm gấp/dễ vỡ/hướng/gần-xa/gây nhiễu/điểm ghé bằng cụm cùng nghĩa, chèn mệnh đề gây nhiễu, đảo thứ
     tự các câu nhỏ.
  2. Câu TỔNG HỢP từ khung câu tự động não (KHUNG_* bên dưới, nguồn: tự soạn, không nhìn validation/test).
Sau đó mọi câu đều có thể bị: bỏ dấu, viết thường, gõ sai (mất 1 chữ / đảo 2 chữ kề — đúng 2 kiểu lỗi của dữ liệu).

Kho cụm từ lấy từ lexicon.py theo chế độ (mù: chỉ cụm có trong train + động não sạch; cuối: toàn bộ).
"pool": "A" / "B" / None — chia đôi mọi kho cụm từ và khung câu (theo mã băm cố định) để làm "dev nội bộ":
học bằng nửa A, đo bằng nửa B (cụm từ + khung câu chưa thấy) mà không cần đụng tới validation.
"""
import random
import re
import unicodedata
import zlib

from . import lexicon as L
from .normalize import normalize, strip_accents

TYPES = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
KINDS = [None, "north", "south", "west", "east", "near", "far",
         "north_most", "south_most", "west_most", "east_most", "anchor_near"]      # 5 loại cuối: V2, mô tả qua bản đồ
MAP_KINDS = {"north_most", "south_most", "west_most", "east_most", "anchor_near"}
WORD_RE = re.compile(r"\w+|[.,!?;:]")
PERSON_HEADS = {"thu", "co", "chu", "anh", "chi", "bac", "thay", "nhan", "can", "nguoi", "quan", "sinh", "ban",
                "truong", "doi", "van", "huan", "hlv", "giao", "giang", "ky", "nghien", "nhom", "tro", "ke", "bao",
                "le", "to", "y", "dieu", "duoc", "dau", "bep", "lop", "hieu", "pho", "clb", "cau"}
PLACE_HEADS = {"noi", "khu", "cho", "day", "phong", "toa", "nha", "san", "quay", "tram", "bai", "ham", "cong",
               "chot", "loi", "van", "xuong", "goc", "diem", "trung", "tu", "kho", "tang", "lan", "bot", "vien",
               "be", "ho", "duong", "lab", "thu", "ktx", "ky", "ki", "can", "cang", "canteen", "cantin", "giang",
               "hoi", "lop", "benh", "y", "parking", "library", "dorm", "gym", "svd", "food", "foodcourt", "lang",
               "barie", "cua", "sanh", "ngoi", "tv"}


def _pool_ok(phrase, pool):
    if pool is None:
        return True
    h = zlib.crc32(normalize(phrase).encode()) % 2
    return (h == 0) == (pool == "A")


def _is_person(alias):
    """Tên gọi qua người nhận (thủ thư, bác sĩ...) hay tên nơi chốn."""
    w = normalize(alias).split()
    persons = ("thu thu", "bac si", "y ta", "dieu duong", "y si", "duoc si", "dau bep", "bep truong", "thu ky",
               "van thu", "ke toan", "giao vu", "bao ve", "le tan", "lop truong", "hieu truong", "giang vien",
               "huan luyen", "hlv", "ky thuat vien", "nghien cuu", "nhom nghien", "sinh vien", "doi ", "van dong",
               "quan ly", "ban quan ly", "nhan vien", "can bo", "co ", "chu ", "anh ", "chi ", "bac ", "thay ",
               "nguoi ", "to ", "tro giang", "cau lac bo", "clb", "truong ", "pho hieu", "ban giam", "giao vien")
    s = " ".join(w) + " "
    return any(s.startswith(p) for p in persons) and not s.startswith(("bac si truc", "to doc"))


def hidden_hit(phrase, exclude):
    """phrase có chứa (theo ranh giới từ) một cụm bị giấu không (thí nghiệm giả lập cách nói mới)."""
    if not exclude:
        return False
    s = " " + normalize(phrase) + " "
    return any(" " + h + " " in s for h in exclude)


class Pools:
    def __init__(self, pool=None, exclude=()):
        from . import brainstorm as BS
        from .mode import FLAG_V2
        self.place, self.person = {}, {}
        for t in TYPES:
            xs = [a for a in dict.fromkeys(L.PLACES[t]) if _pool_ok(a, pool)]
            self.person[t] = [a for a in xs if _is_person(a)]
            self.place[t] = [a for a in xs if not _is_person(a)]
        f = lambda xs: [x for x in dict.fromkeys(xs) if _pool_ok(x, pool) and not hidden_hit(x, exclude)]  # noqa: E731
        self.flag_v2 = FLAG_V2
        self.mat_fra = f(BS.MATERIALS_FRAGILE) if FLAG_V2 else []
        self.mat_sturdy = f(BS.MATERIALS_STURDY) if FLAG_V2 else []
        self.urg_pos, self.urg_neg = f(L.URGENT_POS), f(L.URGENT_NEG)
        self.fra_pos, self.fra_neg = f(L.FRAGILE_POS), f(L.FRAGILE_NEG)
        self.dirs = {k: f(v) for k, v in L.DIRECTIONS.items()}
        self.near, self.far = f(L.NEAR), f(L.FAR)
        self.via = f([x for x in L.VIA if len(normalize(x).split()) >= 1])
        dis = f(L.DISTRACT)
        self.dis_skip = [x for x in dis if re.search(r"(ghe|qua|toi|den|vao)$", normalize(x))
                         and normalize(x).startswith(("khong", "dung", "khoi", "chang", "cha", "ko", "k ", "kh "))]
        self.dis_nham = [x for x in dis if "nham" in normalize(x)]
        self.dis_left = [x for x in dis if re.search(r"\b(roi|khoi|chuyen)\b", normalize(x))
                         and "nham" not in normalize(x)]
        self.dis_done = [x for x in dis if re.search(r"da giao|da nhan|da lay", normalize(x))]
        self.pool = pool


# ----------------------------------------------------------------- khung câu tổng hợp (tự động não)
GREET = ["", "", "Robot ơi, ", "Xin chào! ", "Nhờ bạn nhé: ", "Chào robot, ", "Alo robot, ", "Hi robot, ",
         "Bạn robot ơi, ", "Yêu cầu mới: ", "[Đơn #{n}] ", "Mã đơn {n}: ", "Robot à, ", "Hello, ", "Ê robot, ",
         "Phiền robot chút: ", "Đơn mới nè: ", "Có việc nhờ: "]
CLOSE = ["", "", "", " Cảm ơn!", " Thanks robot.", " Cảm ơn nhiều.", " Tks.", " Cám ơn bạn.", " Cảm ơn nhé.",
         " Thank you!", " Nhờ cả vào bạn.", " Ok nhé."]
ITEMS = ["gói hàng", "thùng hàng", "hộp giấy", "bưu kiện", "tập tài liệu", "phong bì", "túi đồ", "hộp quà",
         "cái laptop", "chồng sách", "lọ hoa", "hộp bánh", "bình nước", "thùng sữa", "cuộn dây điện", "máy tính bảng",
         "bộ hồ sơ", "hộp linh kiện", "chiếc ô", "túi thuốc", "giấy mời", "bản vẽ", "bộ cờ vua", "cái quạt",
         "đôi giày", "chùm chìa khóa", "hộp mực in", "thùng nước", "tờ đơn", "kiện hàng"]
# {G} nơi giao, {I} món hàng; {P} người nhận
GOAL_FRAMES = [
    "Giao {I} tới {G}.", "Mang {I} đến {G} giúp mình.", "Chuyển {I} sang {G}.", "Đem {I} qua {G} nhé.",
    "Phiền bạn đưa {I} tới {G}.", "Đích đến là {G}.", "Điểm giao: {G}.", "Nơi nhận: {G}.", "{G} đang cần {I}.",
    "Có người ở {G} chờ {I}.", "Hàng cần tới {G}.", "Bạn chạy qua {G} đưa {I} nhé.", "Làm ơn giao {I} về {G}.",
    "Gửi tới {G}: {I}.", "Mình cần {I} ở {G}.", "Địa chỉ giao: {G}.", "Đưa giúp {I} về {G}.",
    "Nhận hàng tại {G}.", "{I} này giao ở {G}.", "Bạn mang {I} lên {G} hộ mình.", "Cho {I} tới {G} nha.",
    "Ship {I} tới {G}.", "Giao hàng: {I}, nơi nhận {G}.", "Đơn này đi {G}.", "Vận chuyển {I} đến {G}.",
    "Cần giao {I} ở {G}.", "Đến {G} giao {I}.", "Nhờ robot chở {I} tới {G}.",
]
PERSON_FRAMES = [
    "{P} đang chờ {I}.", "{P} cần nhận {I}.", "Giao {I} cho {P}.", "Đưa {I} tận tay {P}.",
    "{P} nhờ mang {I} tới.", "Người nhận là {P}.", "{I} gửi {P}.", "{P} đang đợi {I}.", "Chuyển {I} cho {P} nhé.",
]
PERSON_AT_FRAMES = ["{P} ở {G} đang chờ {I}.", "{P} tại {G} cần nhận {I}.", "Giao {I} cho {P} ở {G}."]
# {V} điểm ghé, {J} món lấy ở điểm ghé
VIA_FRAMES = [
    "Nhớ ghé {V} lấy {J} trước.", "Trước đó qua {V} lấy {J}.", "Ghé {V} trước đã.", "Tạt vào {V} nhận {J} trước.",
    "Lấy {J} ở {V} trước nhé.", "Phải qua {V} trước.", "Dừng ở {V} lấy đồ trước.", "Đầu tiên ghé {V}.",
    "Ghé ngang {V} lấy {J} trước.", "Trước hết tới {V} lấy {J}.", "Tiện đường ghé {V} lấy {J} trước.",
]
COMBO_FRAMES = [
    "Trước khi mang {I} tới {G}, nhớ ghé {V} lấy {J}.", "Ghé {V} lấy {J} trước, rồi đưa {I} đến {G}.",
    "Trước tiên qua {V} nhận {J}, sau đó chuyển {I} tới {G}.", "Lấy {J} ở {V} xong thì đem {I} sang {G}.",
    "Đi {V} trước, sau cùng tới {G}.", "Giao {I} tới {G} nhưng phải ghé {V} trước.",
    "Bước 1: lấy {J} ở {V}. Bước 2: giao {I} tới {G}.", "Đầu tiên tới {V}, sau đó mới tới {G}.",
    "Tới {G} giao {I}, nhưng trước đó ghé {V} lấy {J}.", "Qua {V} lấy {J} rồi mới mang {I} tới {G}.",
    "Trước khi tới {G}, tạt qua {V} một chút.", "Mang {I} tới {G}; nhớ ghé {V} lấy {J} trước khi đi.",
]
DISTRACT_FRAMES = [
    "Không cần ghé {D}.", "Đừng nhầm với {D}.", "Người nhận đã rời {D} rồi.", "Hôm qua đã giao ở {D} rồi.",
    "{D} không phải nơi nhận.", "Không phải {D} đâu nhé.", "Đừng giao nhầm sang {D}.", "Bỏ qua {D}.",
    "Khỏi ghé {D}.", "Đừng đi {D}.", "Chứ không phải {D}.", "Người nhận không còn ở {D}.",
    "Lần trước đã giao tới {D} rồi.", "Không cần qua {D} đâu.",
]
REF_DIR_FRAMES = ["{X} {d}", "{X} nằm {d}", "{X} ở {d}", "{X} ({d})", "{X} phía {dd}"]
REF_NEAR_FRAMES = ["{X} {n} {A} hơn", "{X} nằm {n} {A}", "{X} {n} {A}", "{X} ở {n} {A}", "{X} {n} {A} nhất"]
FLAG_FRAMES = ["{f}.", "{f} nhé!", "Lưu ý: {f}.", "{f} nha.", "Chú ý: {f}.", "{f}!", "Bên trong là {f}.",
               "Trong hộp có {f}.", "Hàng là {f}.", "Món này {f}.", "Nhắc nhẹ: {f}.", "Nhớ là {f} nhé."]
DIR_SHORT = {"north": ["bắc", "trên"], "south": ["nam", "dưới"], "west": ["tây", "trái"], "east": ["đông", "phải"]}


# ----------------------------------------------------------------- V2: mô tả qua bản đồ, đính chính (tự động não)
# NGUỒN: tự động não (Claude, 08/10/2026), không nhìn validation / test. {H} = danh từ chỉ nơi, {A} = mốc.
MAP_HEADS = ["địa điểm", "chỗ", "nơi", "điểm", "tòa nhà", "khu", "điểm giao", "công trình", "vị trí"]
MAP_FRAMES = {
    "north_most": ["{H} nằm cao nhất trên bản đồ", "{H} ở trên cùng bản đồ", "{H} nằm sát mép trên nhất", "{H} xa nhất về phía bắc",
                   "{H} ở phía bắc nhất", "{H} nằm trên cùng", "{H} cao nhất bản đồ", "{H} nằm ở hàng trên cùng",
                   "{H} gần mép trên bản đồ nhất", "{H} bắc nhất"],
    "south_most": ["{H} nằm thấp nhất trên bản đồ", "{H} ở dưới cùng bản đồ", "{H} nằm sát mép dưới nhất", "{H} xa nhất về phía nam",
                   "{H} ở phía nam nhất", "{H} nằm dưới cùng", "{H} thấp nhất bản đồ", "{H} nằm ở hàng dưới cùng",
                   "{H} gần mép dưới bản đồ nhất", "{H} nam nhất"],
    "west_most": ["{H} nằm bên trái nhất bản đồ", "{H} ngoài cùng bên trái", "{H} nằm sát mép trái nhất", "{H} xa nhất về phía tây",
                  "{H} ở phía tây nhất", "{H} nằm tận cùng bên trái", "{H} trái nhất bản đồ", "{H} nằm ở cột ngoài cùng bên trái",
                  "{H} gần mép trái bản đồ nhất", "{H} tây nhất"],
    "east_most": ["{H} nằm bên phải nhất bản đồ", "{H} ngoài cùng bên phải", "{H} nằm sát mép phải nhất", "{H} xa nhất về phía đông",
                  "{H} ở phía đông nhất", "{H} nằm tận cùng bên phải", "{H} phải nhất bản đồ", "{H} nằm ở cột ngoài cùng bên phải",
                  "{H} gần mép phải bản đồ nhất", "{H} đông nhất"],
    "anchor_near": ["{H} gần {A} nhất", "{H} nằm sát {A} nhất", "{H} cạnh {A} nhất", "{H} kề {A} nhất", "{H} gần với {A} nhất",
                    "{H} nằm kế bên {A} nhất", "{H} ở sát bên {A} nhất", "{H} liền kề {A} nhất", "{H} cách {A} gần nhất",
                    "{H} nằm ngay cạnh {A} nhất", "{H} gần {A} hơn cả", "{H} gần {A} nhất trên bản đồ"],
}
CORRECTION_FRAMES = [
    "Tin trước ghi nhầm là {D}.", "(tin trước ghi {D} là nhầm)", "Đính chính: không phải {D}.", "Sửa lại: không giao {D} nữa.",
    "Hủy đơn giao {D}.", "Không giao {D} nữa nhé, đổi lại như sau.", "Lúc nãy nhắn nhầm thành {D}.", "Bỏ qua {D}, ghi nhầm.",
    "Giao {I} tới không phải {D} mà là {G}.", "Không phải {D} mà là {G} nhé, giao {I} tới đó.",
    "Hủy đơn giao {D}. Thay vào đó: giao {I} tới {G}.", "Đổi lại: giao {I} tới {G} chứ không phải {D}.",
    "Thay vì {D}, hãy giao {I} tới {G}.", "Nhầm rồi, không phải {D}, mà là {G}.", "Đính chính lại: giao {I} ở {G}, không phải ở {D}.",
]


def _split_frames(frames, pool):
    return [f for f in frames if _pool_ok(f, pool)] or frames


# ----------------------------------------------------------------- nhiễu
def strip_all(text):
    return strip_accents(text)


def typo(word, rng):
    m = re.match(r"^(\w+)(\W*)$", word)
    if not m or len(m.group(1)) < 3:
        return word
    word, tail = m.group(1), m.group(2)
    return _typo(word, rng) + tail


def _typo(word, rng):
    i = rng.randrange(len(word) - 1)
    if rng.random() < 0.5:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


# 10/10: đề nói test có "lỗi gõ và câu không dấu xuất hiện nhiều hơn" -> COURIER_NLP_NOISE=2: bỏ dấu 50%, lỗi gõ 65% câu,
# 1-5 lỗi / câu (mặc định 1 = như bản BEST: 35% / 45% / 1-3 lỗi).
NOISE = int(__import__("os").environ.get("COURIER_NLP_NOISE", "1"))
_NZ = {1: (0.35, 0.45, [1, 1, 2, 3]), 2: (0.5, 0.65, [1, 2, 3, 4, 5])}[NOISE]


def noisy(text, rng):
    if rng.random() < _NZ[0]:
        text = strip_all(text)
    if rng.random() < 0.3:
        text = text.lower()
    if rng.random() < _NZ[1]:
        words = text.split(" ")
        for _ in range(rng.choice(_NZ[2])):
            k = rng.randrange(len(words))
            words[k] = typo(words[k], rng)
        text = " ".join(words)
    return text


def join_words(ws):
    out = ""
    for w in ws:
        if w in ".,!?;:" and out:
            out += w
        else:
            out += (" " if out else "") + w
    return out


def shuffle_sentences(text, rng):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    if len(parts) > 2:
        head = parts[0] if re.match(r"^(robot|xin|nh[ờo]|y[êe]u|ch[àa]o|\[)", parts[0], re.I) else None
        rest = parts[1:] if head else parts
        rng.shuffle(rest)
        parts = ([head] if head else []) + rest
    return " ".join(parts)


# ----------------------------------------------------------------- nhãn
def empty_label():
    return {"goal": None, "via": None, "goal_ref": None, "via_ref": None, "urgent": False, "fragile": False}


def label_from_mission(m):
    r = lambda x: None if x is None else {"kind": x["kind"], "anchor": x["anchor"]}  # noqa: E731
    return {"goal": m["goal"], "via": m["via"], "goal_ref": r(m["goal_ref"]), "via_ref": r(m["via_ref"]),
            "urgent": m["urgent"], "fragile": m["fragile"]}


# ----------------------------------------------------------------- biến đổi câu thật
class RealAugmenter:
    def __init__(self, pools):
        from .parse import _item_mask, _phrases
        from .text import correct_tokens, find_places
        self.P = pools
        self._item_mask, self._phrases = _item_mask, _phrases
        self.correct, self.find_places = correct_tokens, find_places
        P = pools
        self.cue_groups = [(self._phrases(x), x) for x in (P.urg_pos, P.urg_neg, P.fra_pos, P.fra_neg, P.near, P.far,
                                                            P.dis_skip, P.dis_nham, P.dis_left, P.dis_done, P.via)
                           if x] + [(self._phrases(v), v) for v in P.dirs.values() if v]

    def split(self, text):
        words = WORD_RE.findall(unicodedata.normalize("NFC", text))
        norm = [normalize(w) for w in words]
        if any(" " in n or not n for n in norm):
            return None
        return words, self.correct(norm)

    def augment(self, text, rng, p_place=0.7, p_cue=0.5):
        sp = self.split(text)
        if sp is None:
            return text
        words, tk = sp
        accented = strip_accents(text) != text
        edits = []                                     # (a, b, new text)
        mask = self._item_mask(tk)
        for a, b, t in self.find_places(tk, mask):
            if rng.random() < p_place:
                src = " ".join(tk[a:b])
                pool = self.P.person[t] if _is_person(src) else self.P.place[t]
                if pool:
                    edits.append((a, b, rng.choice(pool)))
        taken = [False] * len(tk)
        for a, b, _ in edits:
            for k in range(a, b):
                taken[k] = True
        for phrases, raw in self.cue_groups:
            for ph in phrases:
                n = len(ph)
                for i in range(len(tk) - n + 1):
                    if tuple(tk[i:i + n]) == ph and not any(taken[i:i + n]) and rng.random() < p_cue:
                        edits.append((i, i + n, rng.choice(raw)))
                        for k in range(i, i + n):
                            taken[k] = True
        edits.sort()
        out, i = [], 0
        for a, b, new in edits:
            out += words[i:a]
            out += new.split() if accented else strip_accents(new).split()
            i = b
        out += words[i:]
        return join_words(out)


# ----------------------------------------------------------------- câu tổng hợp
class SynthGenerator:
    def __init__(self, pools):
        self.P = pools
        pool = pools.pool
        self.goal_frames = _split_frames(GOAL_FRAMES, pool)
        self.person_frames = _split_frames(PERSON_FRAMES, pool)
        self.person_at = _split_frames(PERSON_AT_FRAMES, pool)
        self.via_frames = _split_frames(VIA_FRAMES, pool)
        self.combo = _split_frames(COMBO_FRAMES, pool)
        self.dis_frames = _split_frames(DISTRACT_FRAMES, pool)
        self.ref_dir = _split_frames(REF_DIR_FRAMES, pool)
        self.ref_near = _split_frames(REF_NEAR_FRAMES, pool)

    def place(self, t, rng):
        return rng.choice(self.P.place[t] or self.P.person[t])

    def ref_text(self, x_text, kind, anchor, rng):
        if kind in ("near", "far"):
            n = rng.choice(self.P.near if kind == "near" else self.P.far)
            return rng.choice(self.ref_near).format(X=x_text, n=n, A=self.place(anchor, rng))
        d = rng.choice(self.P.dirs[kind])
        return rng.choice(self.ref_dir).format(X=x_text, d=d, dd=rng.choice(DIR_SHORT[kind]))

    def make(self, rng):
        lab = empty_label()
        types = list(TYPES)
        rng.shuffle(types)
        g = types.pop()
        lab["goal"] = g
        v = types.pop() if rng.random() < 0.45 else None
        lab["via"] = v
        others = types
        G = self.place(g, rng)
        V = self.place(v, rng) if v else None

        def maybe_ref(field, txt, t):
            r = rng.random()
            if r < 0.22:
                k = rng.choice(["north", "south", "west", "east"])
                lab[field] = {"kind": k, "anchor": None}
                return self.ref_text(txt, k, None, rng)
            if r < 0.4:
                k = rng.choice(["near", "far"])
                a = others.pop() if others else None
                if a:
                    lab[field] = {"kind": k, "anchor": a}
                    return self.ref_text(txt, k, a, rng)
            return txt
        if rng.random() < 0.45:                       # V2: nơi giao CHỈ mô tả qua bản đồ (không nêu loại)
            k = rng.choice(["north_most", "south_most", "west_most", "east_most", "anchor_near", "anchor_near"])
            a = others.pop() if (k == "anchor_near" and others) else None
            if k == "anchor_near" and a is None:
                k = "north_most"
            lab["goal_ref"] = {"kind": k, "anchor": a}
            G = rng.choice(MAP_FRAMES[k]).format(H=rng.choice(MAP_HEADS), A=self.place(a, rng) if a else "")
        else:
            G = maybe_ref("goal_ref", G, g)
        if V:
            V = maybe_ref("via_ref", V, v)
        I, J = rng.choice(ITEMS), rng.choice(ITEMS)
        sents = []
        if v and rng.random() < 0.6:
            sents.append(rng.choice(self.combo).format(G=G, V=V, I=I, J=J))
        else:
            if self.P.person[g] and rng.random() < 0.3 and not lab["goal_ref"]:
                sents.append(rng.choice(self.person_frames).format(P=rng.choice(self.P.person[g]), I=I))
            elif self.P.person[g] and rng.random() < 0.15:
                sents.append(rng.choice(self.person_at).format(P=rng.choice(self.P.person[g]), G=G, I=I))
            else:
                sents.append(rng.choice(self.goal_frames).format(G=G, I=I))
            if v:
                vs = rng.choice(self.via_frames).format(V=V, J=J)
                if rng.random() < 0.5:
                    sents.insert(0, vs)
                else:
                    sents.append(vs)
        if others and rng.random() < 0.3:              # V2: đính chính / đối chiếu — nơi bị gạch bỏ thành gây nhiễu
            D = self.place(others.pop(), rng)
            fr = rng.choice(CORRECTION_FRAMES)
            if "{G}" in fr:
                sents = [fr.format(D=D, G=G, I=I)] + sents[1:] if sents else [fr.format(D=D, G=G, I=I)]
            else:
                sents.insert(rng.randrange(len(sents) + 1), fr.format(D=D, I=I))
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            if others:
                sents.insert(rng.randrange(len(sents) + 1),
                             rng.choice(self.dis_frames).format(D=self.place(others.pop(), rng)))
        for flag, pos, neg, mats in (("urgent", self.P.urg_pos, self.P.urg_neg, None),
                                     ("fragile", self.P.fra_pos, self.P.fra_neg, (self.P.mat_fra, self.P.mat_sturdy))):
            sent, val = flag_sentence(rng, pos, neg, mats, deadline=flag == "urgent" and self.P.flag_v2,
                                      v2=self.P.flag_v2)
            if sent:
                lab[flag] = val
                sents.insert(rng.randrange(len(sents) + 1), sent)
        text = rng.choice(GREET).format(n=rng.randrange(10, 9999)) + " ".join(sents) + rng.choice(CLOSE)
        text = text[0].upper() + text[1:]
        return text, lab


DEADLINE_FRAMES = ["Phải tới trước {h}h.", "Cần có trước {h} giờ.", "Giao trước {h}h nhé.", "Trước {h}h phải có hàng.",
                   "Trong {n} phút nữa là cần.", "Cần trong vòng {n} phút.", "Giao trong {n} phút giúp mình."]
MAT_FRAMES = ["Bên trong là {m}.", "Trong hộp có {m}.", "Hàng là {m}.", "Gói này đựng {m}.", "Thùng có {m} bên trong."]


def flag_sentence(rng, pos, neg, mats=None, deadline=False, v2=False):
    """Một câu nhỏ nói về cờ (gấp / dễ vỡ) và nhãn của nó; (None, False) = không nhắc tới."""
    r = rng.random()
    if r < 0.38 and pos:
        return rng.choice(FLAG_FRAMES).format(f=rng.choice(pos)), True
    if r < 0.62 and neg:
        return rng.choice(FLAG_FRAMES).format(f=rng.choice(neg)), False
    if not v2:
        return None, False
    if r < 0.67 and deadline:
        return rng.choice(DEADLINE_FRAMES).format(h=rng.randrange(7, 18), n=rng.choice([5, 10, 15, 20, 30])), True
    if r < 0.72 and mats and mats[0]:
        return rng.choice(MAT_FRAMES).format(m=rng.choice(mats[0])), True
    if r < 0.75 and mats and mats[1]:
        return rng.choice(MAT_FRAMES).format(m=rng.choice(mats[1])), False
    if r < 0.79:                                     # phủ định kép: "không phải là không gấp đâu" = gấp
        ns = [x for x in neg if normalize(x).startswith("khong ")]
        if ns:
            return f"Không phải là {rng.choice(ns)} đâu.", True
    if r < 0.83 and pos:                             # phủ định cụm khẳng định: "không phải hỏa tốc đâu" = không
        ps = [x for x in pos if not normalize(x).startswith(("khong", "dung", "chang", "cha "))]
        if ps:
            return f"{rng.choice(['Không phải', 'Chẳng phải', 'Không hề'])} {rng.choice(ps)} đâu.", False
    return None, False


def distract_sentence(lab, gen, rng):
    used = {lab["goal"], lab["via"]} | {r["anchor"] for r in (lab["goal_ref"], lab["via_ref"]) if r}
    free = [t for t in TYPES if t not in used]
    return rng.choice(gen.dis_frames).format(D=gen.place(rng.choice(free), rng))


def build_dataset(scenes, n_real=8, n_synth=15000, pool=None, seed=2026, exclude=()):
    """Danh sách (câu, nhãn). scenes: cảnh có mission thật (train, hoặc train + validation ở chế độ cuối).
    exclude: cụm từ bị giấu (thí nghiệm giả lập cách nói mới) — không dùng trong mọi kho thay thế."""
    rng = random.Random(seed)
    P = Pools(pool, exclude)
    aug, gen = RealAugmenter(P), SynthGenerator(P)
    out = []
    for s in scenes:
        text, lab = s["mission"]["text"], label_from_mission(s["mission"])
        out.append((text, lab))
        for _ in range(n_real):
            t = aug.augment(text, rng)
            if rng.random() < 0.3:
                t = shuffle_sentences(t, rng)
            if rng.random() < 0.3:
                ps = re.split(r"(?<=[.!?])\s+", t.strip())
                ps.insert(rng.randrange(len(ps) + 1), distract_sentence(lab, gen, rng))
                t = " ".join(ps)
            out.append((noisy(t, rng), lab))
    for _ in range(n_synth):
        t, lab = gen.make(rng)
        out.append((noisy(t, rng), lab))
    return out
