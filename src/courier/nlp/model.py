"""Mô hình ngôn ngữ học sẵn cho khối 2 (bước 5B): đọc cả câu, trả về xác suất cho 8 đầu ra
(goal, via, loại tham chiếu + mốc của goal, loại tham chiếu + mốc của via, gấp, dễ vỡ).

Bộ mã hóa (chọn một, đều chạy offline sau khi tải về):
  - "visobert": uitnlp/visobert (97,6 triệu tham số; học trên văn bản mạng xã hội tiếng Việt: không dấu, viết tắt,
                gõ sai). Bộ tách từ "nhanh" của transformers 5 đọc SAI file sentencepiece của mô hình này, nên ở đây
                dùng trực tiếp thư viện sentencepiece (đã kiểm tra: mô hình đoán từ bị che hợp lý).
  - "phobert":  vinai/phobert-base (135 triệu, giấy phép MIT).
  - "e5":       intfloat/multilingual-e5-small (118 triệu, giấy phép MIT).
Kết hợp với luật + ràng buộc bản đồ ở joint_decode().
"""
from pathlib import Path

from .augment import TYPES, KINDS, MAP_KINDS

ROOT = Path(__file__).resolve().parents[3]
MODELS = Path(__import__("os").environ.get("COURIER_NLP_DIR") or ROOT / "models")   # 10/10: thư mục mô hình câu thử nghiệm
HEADS = {"goal": 10, "via": 11, "gk": len(KINDS), "ga": 11, "vk": len(KINDS), "va": 11, "urgent": 2, "fragile": 2}
HF_NAME = {"visobert": "uitnlp/visobert", "phobert": "vinai/phobert-base", "e5": "intfloat/multilingual-e5-small"}
# Bản sao cục bộ (chạy offline, không cần mạng / cache HF): models/phobert-base/ (config + tokenizer + trọng số gốc).
LOCAL = {"phobert": Path(__file__).resolve().parents[3] / "models" / "phobert-base"}


def _src(backbone):
    p = LOCAL.get(backbone)
    return str(p) if p is not None and (p / "config.json").exists() else HF_NAME[backbone]
MAX_LEN = 160
# trọng số "phiếu" của luật trong giải mã chung (chọn theo điểm chế độ mù, xem docs); đổi bằng COURIER_NLP_W
DEFAULT_W = float(__import__("os").environ.get("COURIER_NLP_W", "2.0"))


def prep(text):
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def encode_label(lab):
    """Nhãn -> chỉ số cho từng đầu ra. Nơi giao CHỈ mô tả qua bản đồ (kind thuộc MAP_KINDS): câu không nêu loại địa
    điểm, nên đầu ra "goal" được BỎ QUA khi học (-100); loại sẽ lấy từ bản đồ khi dự đoán."""
    t = lambda x: 0 if x is None else TYPES.index(x) + 1  # noqa: E731
    k = lambda r: 0 if r is None else KINDS.index(r["kind"])  # noqa: E731
    a = lambda r: 0 if r is None or not r.get("anchor") else TYPES.index(r["anchor"]) + 1  # noqa: E731
    gmap = lab["goal_ref"] is not None and lab["goal_ref"]["kind"] in MAP_KINDS
    return {"goal": -100 if gmap else TYPES.index(lab["goal"]), "via": t(lab["via"]), "gk": k(lab["goal_ref"]),
            "ga": a(lab["goal_ref"]), "vk": k(lab["via_ref"]), "va": a(lab["via_ref"]), "urgent": int(lab["urgent"]),
            "fragile": int(lab["fragile"])}


class Tok:
    """Bộ tách từ thống nhất: trả về (input_ids, attention_mask) dạng tensor."""

    def __init__(self, backbone):
        self.backbone = backbone
        if backbone == "visobert":
            import sentencepiece as spm
            from huggingface_hub import hf_hub_download
            f = hf_hub_download(HF_NAME[backbone], "sentencepiece.bpe.model", local_files_only=True)
            self.sp = spm.SentencePieceProcessor(model_file=f)
            self.pad = 1
        else:
            from transformers import AutoTokenizer
            self.tk = AutoTokenizer.from_pretrained(_src(backbone), local_files_only=True)
            self.pad = self.tk.pad_token_id

    def __call__(self, texts):
        import torch
        texts = [prep(t) for t in texts]
        if self.backbone == "e5":
            texts = ["query: " + t for t in texts]
        if self.backbone == "visobert":
            ids = [[0] + [i + 1 for i in self.sp.encode(t)][:MAX_LEN - 2] + [2] for t in texts]   # bù lệch fairseq
        else:
            ids = self.tk(texts, truncation=True, max_length=MAX_LEN)["input_ids"]
        n = max(len(x) for x in ids)
        X = torch.full((len(ids), n), self.pad, dtype=torch.long)
        M = torch.zeros((len(ids), n), dtype=torch.long)
        for i, x in enumerate(ids):
            X[i, :len(x)] = torch.tensor(x)
            M[i, :len(x)] = 1
        return X, M


def build_net(backbone):
    import torch
    from transformers import AutoModel

    class MissionNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(_src(backbone), local_files_only=True, add_pooling_layer=False)
            h = self.enc.config.hidden_size
            self.drop = torch.nn.Dropout(0.1)
            self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(2 * h, n) for k, n in HEADS.items()})

        def forward(self, ids, mask):
            hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(hs.dtype)
            z = torch.cat([hs[:, 0], (hs * m).sum(1) / m.sum(1)], -1)
            z = self.drop(z)
            return {k: head(z) for k, head in self.heads.items()}

    return MissionNet()


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------- dự đoán
class Predictor:
    def __init__(self, path, device=None):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.backbone = ck["backbone"]
        self.tok = Tok(self.backbone)
        self.net = build_net(self.backbone)
        self.net.load_state_dict(ck["state"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device).eval()
        self.torch = torch
        self.cache = {}

    def probs(self, texts, bs=64):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        with self.torch.no_grad():
            for i in range(0, len(todo), bs):
                batch = todo[i:i + bs]
                X, M = self.tok(batch)
                out = self.net(X.to(self.device), M.to(self.device))
                P = {k: self.torch.log_softmax(v.float(), -1).cpu().numpy() for k, v in out.items()}
                for j, t in enumerate(batch):
                    self.cache[t] = {k: P[k][j] for k in P}
        return [self.cache[t] for t in texts]


    @classmethod
    def from_net(cls, net, backbone, device):
        self = cls.__new__(cls)
        import torch
        self.backbone, self.tok, self.net, self.device, self.torch, self.cache = (
            backbone, Tok(backbone), net.eval(), device, torch, {})
        return self


FREEZE_EMB = __import__("os").environ.get("COURIER_NLP_FREEZE_EMB", "0") == "1"


def fit(train, backbone, epochs=3, bs=32, lr=3e-5, seed=2026, log=print):
    """Huấn luyện MissionNet trên danh sách (câu, nhãn). Câu cùng độ dài được gom chung lô (học nhanh hơn);
    thứ tự lô xáo trộn theo seed cố định."""
    import random
    import time
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tok(backbone)
    net = build_net(backbone).to(device)
    if FREEZE_EMB:      # giữ nguyên nghĩa học sẵn của từng từ: từ chưa gặp khi học vẫn "gần" từ đồng nghĩa đã gặp
        net.enc.embeddings.word_embeddings.weight.requires_grad_(False)
    enc_p = [p for n, p in net.named_parameters() if n.startswith("enc.") and p.requires_grad]
    head_p = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": lr}, {"params": head_p, "lr": 1e-3}], weight_decay=0.01)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(0.06 * steps)))
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(seed)
    lens = [len(t) for t, _ in train]
    step, t0 = 0, time.time()
    net.train()
    for ep in range(epochs):
        idx = list(range(len(train)))
        rng.shuffle(idx)
        batches = []
        for i in range(0, len(idx), bs * 50):            # gom theo độ dài trong từng khối 50 lô
            chunk = sorted(idx[i:i + bs * 50], key=lambda j: lens[j])
            batches += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
        rng.shuffle(batches)
        tot = 0.0
        for b in batches:
            X, M = tok([train[j][0] for j in b])
            Y = [encode_label(train[j][1]) for j in b]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = net(X.to(device), M.to(device))
                loss = sum(ce(out[k].float(), torch.tensor([y[k] for y in Y], device=device)) for k in HEADS)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            tot += loss.item()
            if step % 200 == 0:
                log(f"  ep {ep + 1} bước {step}/{steps} loss {tot / 200:.4f}  ({time.time() - t0:.0f} giây)")
                tot = 0.0
    return net.eval(), tok, device


_PRED = "default"


def model_path():
    from .mode import MODE, TAG
    return MODELS / f"nlp_{MODE}{TAG}.pt"


def get_predictor():
    """Mô hình của chế độ hiện tại (models/nlp_<mode>.pt); None nếu tắt hoặc chưa huấn luyện."""
    global _PRED
    if _PRED == "default":
        from .mode import MODEL
        p = model_path()
        _PRED = Predictor(p) if MODEL and p.exists() else None
    return _PRED


def set_predictor(pred):
    global _PRED
    _PRED = pred


# ----------------------------------------------------------------- giải mã chung: mô hình + luật + bản đồ
def _rule_view(parsed, landmarks):
    from .mode import CONSTRAINTS  # noqa: F401
    from .resolve import apply_constraints
    p = {k: v for k, v in parsed.items() if k != "_probs"}
    return apply_constraints(p, landmarks)


def joint_decode(parsed, landmarks, w=None):
    """Chọn phương án có điểm cao nhất thỏa ràng buộc bản đồ.
    điểm = log xác suất của mô hình + w nếu trùng với kết quả của luật (luật + ràng buộc).
    w = 0: chỉ mô hình; w rất lớn: luật trước, mô hình chỉ quyết khi luật bỏ trống."""
    w = DEFAULT_W if w is None else w
    P = parsed["_probs"]
    R = _rule_view(parsed, landmarks) if landmarks else parsed
    cnt = {t: len(landmarks.get(t, [])) for t in TYPES} if landmarks else {t: 1 for t in TYPES}
    on = [t for t in TYPES if cnt[t] > 0] or TYPES
    bonus = lambda c, r: w if c == r else 0.0  # noqa: E731

    # V2: nơi giao chỉ mô tả qua bản đồ? (so xác suất các loại tham chiếu "bản đồ" với các loại còn lại)
    old_k = [k for k in KINDS if k not in MAP_KINDS]
    k_map = max(MAP_KINDS, key=lambda k: (P["gk"][KINDS.index(k)], k))
    if landmarks and P["gk"][KINDS.index(k_map)] > max(P["gk"][KINDS.index(k)] for k in old_k):
        rc, a = resolve_map_goal(k_map, P["ga"], landmarks)
        if rc is not None:
            goal = next(t for t, xs in landmarks.items() if rc in [tuple(x) for x in xs])
            vias = [None] + [t for t in on if t != goal]
            vidx = lambda t: 0 if t is None else TYPES.index(t) + 1  # noqa: E731
            via = max(vias, key=lambda t: (P["via"][vidx(t)] + bonus(t, R["via"]), -vidx(t)))
            out = dict(parsed)
            out["goal"], out["via"] = goal, via
            out["goal_ref"] = {"kind": k_map, "anchor": a, "rc": rc}
            out["via_ref"] = _ref_old(P, via, "vk", "va", R["via_ref"] if R["via"] == via else None, cnt, bonus)
            _flags(out, parsed, P, R, bonus)
            return out
    goal = max(on, key=lambda t: (P["goal"][TYPES.index(t)] + bonus(t, R["goal"]), -TYPES.index(t)))
    vias = [None] + [t for t in on if t != goal]
    vidx = lambda t: 0 if t is None else TYPES.index(t) + 1  # noqa: E731
    via = max(vias, key=lambda t: (P["via"][vidx(t)] + bonus(t, R["via"]), -vidx(t)))

    out = dict(parsed)
    out["goal"], out["via"] = goal, via
    out["goal_ref"] = _ref_old(P, goal, "gk", "ga", R["goal_ref"] if R["goal"] == goal else None, cnt, bonus)
    out["via_ref"] = _ref_old(P, via, "vk", "va", R["via_ref"] if R["via"] == via else None, cnt, bonus)
    _flags(out, parsed, P, R, bonus)
    return out


def _ref_old(P, x, kh, ah, rref, cnt, bonus):
    """Tham chiếu kiểu cũ (bắc/nam/tây/đông, gần/xa X) cho loại x có 2 bản."""
    if x is None or cnt.get(x, 0) != 2:
        return None
    rk = rref["kind"] if rref else None
    ra = rref.get("anchor") if rref else None
    uniq = [t for t in TYPES if cnt[t] == 1 and t != x]
    kinds = [k for k in KINDS if k not in MAP_KINDS and (k not in ("near", "far") or uniq)]
    k = max(kinds, key=lambda k: (P[kh][KINDS.index(k)] + bonus(k, rk), -KINDS.index(k)))
    if k is None:
        return None
    if k in ("near", "far"):
        a = max(uniq, key=lambda t: (P[ah][TYPES.index(t) + 1] + bonus(t, ra), -TYPES.index(t)))
        return {"kind": k, "anchor": a}
    return {"kind": k, "anchor": None}


def resolve_map_goal(kind, p_anchor, landmarks):
    """Nơi giao mô tả qua bản đồ -> (ô, mốc). Định nghĩa khớp 100% goal_ref.rc trên train V2:
    *_most: xét MỌI địa điểm, hàng nhỏ nhất / lớn nhất, cột nhỏ nhất / lớn nhất;
    anchor_near: mốc = loại có đúng 1 bản (chọn theo mô hình); địa điểm (trừ mốc) có khoảng cách MANHATTAN nhỏ nhất."""
    allp = sorted({tuple(x) for xs in landmarks.values() for x in xs})
    if not allp:
        return None, None
    if kind == "anchor_near":
        uniq = [t for t in TYPES if len(landmarks.get(t, [])) == 1]
        if not uniq:
            return None, None
        a = max(uniq, key=lambda t: (p_anchor[TYPES.index(t) + 1], -TYPES.index(t)))
        x = tuple(landmarks[a][0])
        cand = [p for p in allp if p != x]
        if not cand:                     # bản đồ đọc ra chỉ có đúng 1 địa điểm (ảnh rất xấu): không giải được
            return None, None
        return min(cand, key=lambda p: (abs(p[0] - x[0]) + abs(p[1] - x[1]), p)), a
    key = {"north_most": lambda p: (p[0], p[1]), "south_most": lambda p: (-p[0], p[1]),
           "west_most": lambda p: (p[1], p[0]), "east_most": lambda p: (-p[1], p[0])}[kind]
    return min(allp, key=key), None


def _flags(out, parsed, P, R, bonus):
    for f in ("urgent", "fragile"):
        out[f] = bool(P[f][1] + bonus(True, R[f]) > P[f][0] + bonus(False, R[f]))
    if "_rule_flags" in parsed and FLAG_VOTE != "model":
        for f in ("urgent", "fragile"):
            args = (bool(P[f][1] > P[f][0]), parsed["_rule_flags"][f], parsed.get("_char_flags", {}).get(f))
            out[f] = prio_flag(*args) if FLAG_VOTE == "prio" else vote_flag(*args)


# Bước 5C: kết hợp ba nguồn cho gấp / dễ vỡ (chọn bằng thí nghiệm giả lập cách nói mới, scripts/flag_novel_exp.py):
#   "prio" (MẶC ĐỊNH) = luật nhận ra cụm nào -> theo luật; không thì n-gram ký tự nhận ra -> theo n-gram;
#                       cả hai "không thấy gì" -> PhoBERT quyết. ("không thấy gì" KHÔNG còn bị coi là "không".)
#   "vote" = biểu quyết 3 nguồn (nguồn "không thấy gì" không bỏ phiếu; hòa -> PhoBERT);
#   "model" = như bước 5B: PhoBERT + phiếu W của luật.
FLAG_VOTE = __import__("os").environ.get("COURIER_FLAG_VOTE", "prio")


def prio_flag(m, r, c):
    for v in (r, c):
        if v is not None:
            return v
    return m


def vote_flag(m, r, c):
    votes = [v for v in (m, r, c) if v is not None]
    t, f = sum(votes), len(votes) - sum(votes)
    return m if t == f else t > f


def lr_lambda(total, warm):
    return lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))

