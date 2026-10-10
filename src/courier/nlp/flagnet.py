"""Bước 5C (phần A): mô hình NHỎ đọc gấp / dễ vỡ theo TỪNG CÂU NHỎ — nguồn thứ ba để biểu quyết với PhoBERT và luật.

TF-IDF n-gram ký tự (2..5) + hồi quy logistic, 3 lớp cho mỗi cờ: "pos" (có), "neg" (phủ định), "none" (không nhắc).
Học từ:
  - các câu nhỏ của câu train (chế độ cuối: + validation), nhãn từng câu nhỏ lấy bằng luật (đúng 100% trên train);
  - câu nhỏ tổng hợp từ kho cụm từ gấp / dễ vỡ (động não) và khung câu, có bỏ dấu và gõ sai.
Tự học lại mỗi lần chạy (vài giây), tất định (seed cố định).
"""
import random
import re

import numpy as np

from .normalize import normalize

SEED = 2026
SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
THRESHOLD = 0.6


def sentences(text):
    return [p for p in SPLIT_RE.split(normalize(text)) if p.strip()]


class FlagNet:
    def __init__(self, X, Y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        A = self.vec.fit_transform([" " + x + " " for x in X])
        self.clf = {}
        for f in ("urgent", "fragile"):
            c = LogisticRegression(C=5, max_iter=3000, random_state=SEED)
            c.fit(A, [y[f] for y in Y])
            self.clf[f] = c

    def n_params(self):
        return sum(c.coef_.size + c.intercept_.size for c in self.clf.values())

    def predict_mission(self, text):
        """-> {"urgent": True/False/None, "fragile": ...} (None = không câu nhỏ nào đủ chắc)."""
        sents = sentences(text)
        if not sents:
            return {"urgent": None, "fragile": None}
        A = self.vec.transform([" " + s + " " for s in sents])
        out = {}
        for f, c in self.clf.items():
            P = c.predict_proba(A)
            cls = list(c.classes_)
            pos = P[:, cls.index("pos")] if "pos" in cls else np.zeros(len(sents))
            neg = P[:, cls.index("neg")] if "neg" in cls else np.zeros(len(sents))
            if (neg >= THRESHOLD).any():
                out[f] = False
            elif (pos >= THRESHOLD).any():
                out[f] = True
            else:
                out[f] = None
        return out


def _typo(s, rng):
    w = s.split()
    k = rng.randrange(len(w))
    x = w[k]
    if len(x) >= 3:
        i = rng.randrange(len(x) - 1)
        w[k] = x[:i] + x[i + 1:] if rng.random() < 0.5 else x[:i] + x[i + 1] + x[i] + x[i + 2:]
    return " ".join(w)


def build_training(scenes, pools, n_synth=12000, seed=SEED):
    """Mẫu học: (câu nhỏ đã chuẩn hóa, {"urgent": pos/neg/none, "fragile": ...})."""
    from .augment import flag_sentence, SynthGenerator
    from .parse import parse_tokens, flag_tri, URG_NEG, URG_POS, FRA_NEG, FRA_POS
    rng = random.Random(seed)
    lab = lambda v: "none" if v is None else ("pos" if v else "neg")  # noqa: E731
    X, Y = [], []
    for s in scenes:
        for sent in SPLIT_RE.split(s["mission"]["text"]):
            if not sent.strip():
                continue
            tk = parse_tokens(sent)
            y = {"urgent": lab(flag_tri(tk, URG_NEG, URG_POS, deadline=True)),
                 "fragile": lab(flag_tri(tk, FRA_NEG, FRA_POS))}
            X.append(normalize(sent))
            Y.append(y)
    gen = SynthGenerator(pools)
    for _ in range(n_synth):
        y = {"urgent": "none", "fragile": "none"}
        f = rng.choice(["urgent", "fragile", "none"])
        if f == "urgent":
            sent, v = flag_sentence(rng, pools.urg_pos, pools.urg_neg, None, deadline=pools.flag_v2, v2=pools.flag_v2)
        elif f == "fragile":
            sent, v = flag_sentence(rng, pools.fra_pos, pools.fra_neg, (pools.mat_fra, pools.mat_sturdy),
                                    v2=pools.flag_v2)
        else:
            sent, v = None, None
        if sent is None:                              # một câu nhỏ bất kỳ của câu tổng hợp, nhãn theo luật
            t, _ = gen.make(rng)
            sent = rng.choice(SPLIT_RE.split(t))
            tk = parse_tokens(sent)
            y = {"urgent": lab(flag_tri(tk, URG_NEG, URG_POS, deadline=True)),
                 "fragile": lab(flag_tri(tk, FRA_NEG, FRA_POS))}
        else:
            y[f] = "pos" if v else "neg"
        x = normalize(sent)
        if rng.random() < 0.4:
            x = _typo(x, rng)
        X.append(x)
        Y.append(y)
    return X, Y


_NET = "default"


def get_flagnet():
    global _NET
    if _NET == "default":
        from .mode import FLAG_V2, MODE
        from . import aux
        if not FLAG_V2:
            _NET = None
        elif aux.get("flagnet") is not None:
            _NET = aux.get("flagnet")
        else:
            from ..data import load_split
            from .augment import Pools
            _, _, sc = load_split("train")
            if MODE == "final":
                from .mode import final_val_scenes
                sc = sc + final_val_scenes()
            _NET = FlagNet(*build_training(sc, Pools()))
    return _NET


def set_flagnet(net):
    global _NET
    _NET = net
