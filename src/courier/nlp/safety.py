"""Lưới an toàn cho khối 2: đoán loại địa điểm cho cụm từ LẠ (chưa có trong từ điển).

Chỉ can thiệp ở "ô trống": vị trí trong câu mà ở đó chắc chắn có một địa điểm (sau "tới/đến/qua/ở/ghé...",
trước "cần nhận/đang chờ", sau "điểm giao:", sau "gần/xa"), nhưng bộ luật KHÔNG nhận ra địa điểm nào.
Cụm từ trong ô được một mô hình nhỏ phân loại thành 1 trong 10 loại, hoặc "none" (không phải địa điểm).

Các phương án (so sánh trong scripts/novel_names_exp.py):
  - "char": TF-IDF n-gram ký tự + hồi quy logistic (vài chục nghìn tham số).
  - "e5":   câu nhúng của mô hình đa ngôn ngữ intfloat/multilingual-e5-small (~118 triệu tham số)
            + hồi quy logistic.
"""
import random

import numpy as np

from .text import find_places, is_typo_of  # noqa: F401

SEED = 2026
NONE = "none"

PREPS = {"toi", "den", "qua", "o", "tai", "ve", "sang", "ghe", "vao"}
ANCHOR_WORDS = {"gan", "xa", "sat", "canh"}
STOP = {".", ",", "!", "?", ";", ":", "giup", "lay", "nhan", "truoc", "nam", "gan", "xa", "sat", "phia", "man",
        "ben", "dang", "can", "xong", "roi", "hon", "va", "nhe", "thi", "sau", "o", "minh",
        "khong", "da", "giao", "mang", "dua", "gui", "chuyen", "van", "la", "hang", "cam", "nho", "voi"}
PERSON_TAIL = [("can", "nhan"), ("dang", "cho")]
MAX_SPAN = 5


def slot_spans(tk, used):
    """Các ô (đầu, cuối) có thể chứa tên địa điểm mà chưa ai nhận ra. used[i] = từ i đã thuộc địa điểm/món hàng."""
    out = []

    def grab(j):
        k = j
        while k < len(tk) and k < j + MAX_SPAN and not used[k] and tk[k] not in STOP and tk[k].isalpha():
            k += 1
        return (j, k) if k > j else None

    for i, w in enumerate(tk):
        nxt = i + 1
        if w in PREPS or w in ANCHOR_WORDS:
            if w == "o" and i > 0 and tk[i - 1] in ("khong", "phai"):     # "không ở", "phải ở đó"
                continue
            if nxt < len(tk) and not used[nxt]:
                sp = grab(nxt)
                if sp:
                    out.append(sp)
        if w == ":" and i >= 2 and tk[i - 2:i] in (["diem", "giao"], ["den", "la"]):
            sp = grab(nxt)
            if sp:
                out.append(sp)
        if w == "la" and i >= 2 and tk[i - 2:i] == ["dich", "den"]:
            sp = grab(nxt)
            if sp:
                out.append(sp)
    # người nhận đứng đầu mệnh đề: "<X> cần nhận / đang chờ"
    for i in range(len(tk) - 1):
        if (tk[i], tk[i + 1]) in PERSON_TAIL:
            j = i
            while j > 0 and i - j < MAX_SPAN and not used[j - 1] and tk[j - 1].isalpha() and tk[j - 1] not in STOP:
                j -= 1
            if j < i:
                out.append((j, i))
    # bỏ ô nằm sát ngay trước một địa điểm đã biết (vd "phòng" trong "phòng lab" khi chỉ biết "lab")
    out = [(a, b) for a, b in set(out) if not (b < len(tk) and used[b] and tk[b].isalpha())]
    return sorted(out)


# ----------------------------------------------------------------- dữ liệu huấn luyện cho bộ đoán loại
def _typo_variants(s, rng, n=2):
    out = []
    words = s.split()
    for _ in range(n):
        w = list(words)
        k = rng.randrange(len(w))
        x = w[k]
        if len(x) >= 3:
            i = rng.randrange(len(x) - 1)
            x = x[:i] + x[i + 1:] if rng.random() < 0.5 else x[:i] + x[i + 1] + x[i] + x[i + 2:]
            w[k] = x
            out.append(" ".join(w))
    return out


def build_training(train_token_lists, aliases, item_mask_fn):
    """Mẫu dương: các tên gọi đã biết (+ biến thể gõ sai). Mẫu âm ("none"): cụm trong ô trống của câu train
    mà không phải tên địa điểm (vd 'do', 'buu kien', 'ban do')."""
    rng = random.Random(SEED)
    X, y = [], []
    for al, typ in aliases:
        s = " ".join(al)
        for v in [s] + _typo_variants(s, rng):
            X.append(v)
            y.append(typ)
    neg = set()
    for tk in train_token_lists:
        mask = item_mask_fn(tk)
        places = find_places(tk, mask)
        used = list(mask)
        for a, b, _ in places:
            for k in range(a, b):
                used[k] = True
        for a, b in slot_spans(tk, used):
            neg.add(" ".join(tk[a:b]))
        for i in range(len(tk)):          # thêm các mảnh ngắn bất kỳ không chạm địa điểm
            if not used[i] and tk[i].isalpha():
                neg.add(tk[i])
    for s in sorted(neg):
        X.append(s)
        y.append(NONE)
    return X, y


# ----------------------------------------------------------------- các phương án
class CharNet:
    """TF-IDF n-gram ký tự (2..5, trong từng từ) + hồi quy logistic."""

    def __init__(self, X, y, threshold=0.5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
        A = self.vec.fit_transform([" " + x + " " for x in X])
        self.clf = LogisticRegression(C=10, max_iter=3000, random_state=SEED)
        self.clf.fit(A, y)
        self.threshold = threshold

    def n_params(self):
        return self.clf.coef_.size + self.clf.intercept_.size

    def predict(self, spans):
        if not spans:
            return []
        P = self.clf.predict_proba(self.vec.transform([" " + s + " " for s in spans]))
        out = []
        for p in P:
            k = int(np.argmax(p))
            lab = str(self.clf.classes_[k])
            out.append(lab if lab != NONE and p[k] >= self.threshold else None)
        return out


class E5Net:
    """Nhúng cụm từ bằng intfloat/multilingual-e5-small (đa ngôn ngữ, 118M tham số) + hồi quy logistic."""
    MODEL = "intfloat/multilingual-e5-small"

    def __init__(self, X, y, threshold=0.5):
        import torch
        from sklearn.linear_model import LogisticRegression
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(SEED)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModel.from_pretrained(self.MODEL).eval()
        self.cache = {}
        E = self._embed(X)
        self.clf = LogisticRegression(C=10, max_iter=3000, random_state=SEED)
        self.clf.fit(E, y)
        self.threshold = threshold

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters()) + self.clf.coef_.size

    def _embed(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        with self.torch.no_grad():
            for i in range(0, len(todo), 256):
                batch = todo[i:i + 256]
                enc = self.tok(["query: " + t for t in batch], padding=True, truncation=True, max_length=32,
                               return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
                e = self.torch.nn.functional.normalize(e, dim=-1)
                for t, v in zip(batch, e.numpy()):
                    self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])

    def predict(self, spans):
        if not spans:
            return []
        P = self.clf.predict_proba(self._embed(spans))
        out = []
        for p in P:
            k = int(np.argmax(p))
            lab = str(self.clf.classes_[k])
            out.append(lab if lab != NONE and p[k] >= self.threshold else None)
        return out


class EnsembleNet:
    """Trung bình xác suất của CharNet và E5Net."""

    def __init__(self, X, y, threshold=0.5):
        self.a, self.b = CharNet(X, y, threshold), E5Net(X, y, threshold)
        assert list(self.a.clf.classes_) == list(self.b.clf.classes_)
        self.threshold = threshold

    def n_params(self):
        return self.a.n_params() + self.b.n_params()

    def predict(self, spans):
        if not spans:
            return []
        P = (self.a.clf.predict_proba(self.a.vec.transform([" " + s + " " for s in spans]))
             + self.b.clf.predict_proba(self.b._embed(spans))) / 2
        out = []
        for p in P:
            k = int(np.argmax(p))
            lab = str(self.a.clf.classes_[k])
            out.append(lab if lab != NONE and p[k] >= self.threshold else None)
        return out


DEFAULT_KIND = "char"
DEFAULT_THRESHOLD = 0.6     # chọn bằng thí nghiệm "giả lập tên mới" (train + val-A)


def build_default_net():
    """Cấu hình chuẩn: CharNet học từ từ điển + câu train (chế độ mù) hoặc train + toàn bộ validation (chế độ
    cuối). Không bao giờ dùng test."""
    from ..data import load_split
    from .mode import MODE
    _, _, tr = load_split("train")
    if MODE == "final":
        from .mode import final_val_scenes
        tr = tr + final_val_scenes()
    return build_net(DEFAULT_KIND, train_scenes=tr, threshold=DEFAULT_THRESHOLD)


def build_net(kind, train_scenes=None, visible_aliases=None, threshold=DEFAULT_THRESHOLD):
    from .parse import parse_tokens, _item_mask
    from .text import current_aliases
    if train_scenes is None:
        from ..data import load_split
        _, _, train_scenes = load_split("train")
    if visible_aliases is None:
        visible_aliases = current_aliases()
    toks = [parse_tokens(s["mission"]["text"]) for s in train_scenes]
    X, y = build_training(toks, visible_aliases, _item_mask)
    if kind == "char":
        return CharNet(X, y, threshold)
    if kind == "e5":
        return E5Net(X, y, threshold)
    if kind == "ens":
        return EnsembleNet(X, y, threshold)
    raise ValueError(kind)
