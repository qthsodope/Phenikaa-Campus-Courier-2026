"""Huấn luyện các mạng của khối đọc ảnh từ đầu (seed cố định). Chỉ dùng ảnh TRAIN (+ val-A nếu --use-val-a).

    .venv/bin/python scripts/train_vision.py --stage kp     --epochs 30   # mạng dò điểm -> models/kp.pt
    .venv/bin/python scripts/train_vision.py --stage patch  --epochs 20   # 3 mạng mảnh ảnh -> models/{edge,node,misc}.pt
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courier.data import load_split, DATA  # noqa: E402
from courier.splits import val_halves  # noqa: E402
from courier.vision import common as C  # noqa: E402
from courier.vision.models import KeypointNet, edge_net, node_net, misc_net, n_params  # noqa: E402

MODELS = ROOT / "models"
SEED = 2026
ROT_MAP = [2, 3, 1, 0]   # hướng mũi sau khi xoay mảnh 90° ngược chiều kim đồng hồ (UP->LEFT, DOWN->RIGHT, ...)


class SceneDS(Dataset):
    def __init__(self, items, augment=True, epoch_seed=0):
        self.items = items          # list of (split_dir, scene)
        self.augment = augment
        self.epoch_seed = epoch_seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, s = self.items[i]
        img = C.load_rgb(d / s["image"])
        t = C.scene_targets(s)
        pts = list(t["node_xy"]) + [(x, y) for x, y, _ in t["legend"]] + [t["weather_xy"]]
        if self.augment:
            rng = random.Random(hash((SEED, self.epoch_seed, i)) & 0xFFFFFFFF)
            img, pts = C.augment(img, pts, rng)
        else:
            pts = np.asarray(pts, dtype=np.float64)
        n = len(t["node_xy"])
        t["node_xy"] = pts[:n]
        t["legend"] = [(pts[n + k][0], pts[n + k][1], t["legend"][k][2]) for k in range(3)]
        t["weather_xy"] = tuple(pts[n + 3])
        return np.asarray(img, dtype=np.uint8), t


def collate(b):
    return b


def oversample(items, k):
    """10/10: lặp lại ảnh lưới lớn (đề: lưới 5-9, train chỉ 4% lưới 9; validation 29%) — lưới 9: k lần, lưới 8: (k+1)//2."""
    if k <= 1:
        return items
    out = []
    for it in items:
        g = max(it[1]["grid"]["rows"], it[1]["grid"]["cols"])
        out += [it] * (k if g >= 9 else (k + 1) // 2 if g == 8 else 1)
    return out


def load_items(use_val_a, use_val_all=False):
    items = []
    _, _, tr = load_split("train")
    items += [(DATA / "train", s) for s in tr]
    _, _, va = load_split("validation")
    A, _ = val_halves(len(va))
    vA = [(DATA / "validation", va[i]) for i in A]
    if use_val_all:                 # bản nộp cuối: train + TOÀN BỘ validation (ảnh "khó" giống test)
        items += [(DATA / "validation", s) for s in va]
    elif use_val_a:
        items += vA
    return items, vA


def img_tensor(a, dev):
    t = torch.from_numpy(np.ascontiguousarray(a).copy()).to(dev).permute(2, 0, 1).float() / 255.0
    return (t - C.MEAN.to(dev)) / C.STD.to(dev)


# ----------------------------------------------------------------- giai đoạn 1: mạng dò điểm
def kp_batch(batch, dev):
    S, R = C.KP_INPUT, C.KP_INPUT // C.KP_STRIDE
    X = torch.zeros(len(batch), 3, S, S, device=dev)
    Y = torch.zeros(len(batch), len(C.KP_CHANNELS), R, R, device=dev)
    yy, xx = torch.meshgrid(torch.arange(R, device=dev).float(), torch.arange(R, device=dev).float(), indexing="ij")
    for k, (a, t) in enumerate(batch):
        H, W = a.shape[:2]
        sc = S / max(H, W)
        im = img_tensor(a, dev)[None]
        im = F.interpolate(im, size=(round(H * sc), round(W * sc)), mode="bilinear", align_corners=False)
        X[k, :, :im.shape[2], :im.shape[3]] = im[0]
        groups = [(0, t["node_xy"])] + [(1 + j, [t["legend"][j][:2]]) for j in range(3)] + [(4, [t["weather_xy"]])]
        for ch, pts in groups:
            for (x, y) in pts:
                cx, cy = x * sc / C.KP_STRIDE - 0.5, y * sc / C.KP_STRIDE - 0.5
                g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2))
                Y[k, ch] = torch.maximum(Y[k, ch], g)
    return X, Y


def focal_loss(logits, Y):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = Y.ge(0.99).float()
    neg = 1 - pos
    pl = -torch.log(p) * (1 - p) ** 2 * pos
    nl = -torch.log(1 - p) * p ** 2 * (1 - Y) ** 4 * neg
    return (pl.sum() + nl.sum()) / pos.sum().clamp(min=1)


def train_kp(a, dev):
    items, vA = load_items(a.use_val_a, a.use_val_all)
    items = oversample(items, a.big_rep)
    model = KeypointNet(backbone=a.backbone)
    if a.init_from:                                   # học tiếp từ trọng số có sẵn (cùng kiến trúc)
        model.load_state_dict(torch.load(Path(a.init_from) / "kp.pt", map_location="cpu", weights_only=True))
    model = model.to(dev).to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler()
    print("KeypointNet tham số:", n_params(model))
    lr = a.lr or 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * (len(items) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        model.train()
        dl = DataLoader(SceneDS(items, True, ep), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                        collate_fn=collate, drop_last=True, generator=torch.Generator().manual_seed(SEED + ep))
        t0, tot = time.time(), 0.0
        for batch in dl:
            X, Y = kp_batch(batch, dev)
            X = X.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(X)
            loss = focal_loss(out.float(), Y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        print(f"[kp] epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out / "kp.pt")


# ----------------------------------------------------------------- giai đoạn 2: mạng mảnh ảnh
def patch_batch(batch, dev, train=True, rng=None):
    E, N, M = {"x": [], "look": [], "stairs": [], "oneway": []}, {"x": [], "content": [], "heading": [],
                                                                  "hmask": []}, {"leg": [], "legy": [], "wx": [], "wy": []}
    for a, t in batch:
        im = img_tensor(a, dev)
        idx = {rc: k for k, rc in enumerate(t["node_rc"])}
        xy = torch.tensor(np.asarray(t["node_xy"]), dtype=torch.float32, device=dev)
        jit = (lambda n: torch.randn(n, 2, device=dev) * 2.0) if train else (lambda n: 0)
        # --- cạnh
        pa = torch.tensor([idx[p[0]] for p in t["pairs"]], device=dev)
        pb = torch.tensor([idx[p[1]] for p in t["pairs"]], device=dev)
        lab = torch.tensor([p[2] for p in t["pairs"]], device=dev)
        A_, B_ = xy[pa] + jit(len(pa)), xy[pb] + jit(len(pb))
        ow = lab[:, 2].clone()
        if train:                                      # đảo chiều ngẫu nhiên: a<->b, nhãn một chiều 1<->2
            flip = torch.rand(len(pa), device=dev) < 0.5
            A_, B_ = torch.where(flip[:, None], B_, A_), torch.where(flip[:, None], A_, B_)
            ow = torch.where(flip & (ow > 0), 3 - ow, ow)
        E["x"].append(C.edge_patches(im, A_, B_))
        E["look"].append(lab[:, 0])
        E["stairs"].append(lab[:, 1])
        E["oneway"].append(ow)
        # --- giao lộ
        cont = torch.tensor(t["content"], device=dev)
        if train:                                      # giữ mọi địa điểm/robot + 12 giao lộ trống ngẫu nhiên
            keep = cont.gt(0) | (torch.rand(len(cont), device=dev) < 12.0 / max(len(cont), 1))
            cont, xy = cont[keep], xy[keep]
        P = C.square_patches(im, xy + jit(len(xy)), C.NODE_REGION, C.NODE_SIZE)
        head = torch.full_like(cont, t["heading"])
        hmask = cont.eq(len(C.NODE_CLASSES) - 1)
        N["x"].append(P)
        N["content"].append(cont)
        N["heading"].append(head)
        N["hmask"].append(hmask)
        if train:                                      # robot: thêm 3 bản xoay 90° (đổi nhãn hướng)
            rp = P[hmask]
            h = t["heading"]
            for k in range(1, 4):
                rp = torch.rot90(rp, 1, dims=(2, 3))
                h = ROT_MAP[h]
                N["x"].append(rp)
                N["content"].append(cont[hmask])
                N["heading"].append(torch.full((len(rp),), h, device=dev))
                N["hmask"].append(torch.ones(len(rp), dtype=torch.bool, device=dev))
        # --- chú giải + thời tiết
        lc = torch.tensor([l[:2] for l in t["legend"]], dtype=torch.float32, device=dev)
        M["leg"].append(C.square_patches(im, lc + jit(3), C.LEG_REGION, C.MISC_SIZE))
        M["legy"].append(torch.tensor([l[2] for l in t["legend"]], device=dev))
        wc = torch.tensor([t["weather_xy"]], dtype=torch.float32, device=dev)
        M["wx"].append(C.square_patches(im, wc + jit(1), C.WEATHER_REGION, C.MISC_SIZE))
        M["wy"].append(torch.tensor([t["rain"]], device=dev))
    cat = lambda d: {k: torch.cat(v) for k, v in d.items()}  # noqa: E731
    return cat(E), cat(N), cat(M)


def patch_losses(nets, E, N, M):
    en, nn_, mn = nets
    oe = en(E["x"])
    le = F.cross_entropy(oe["look"], E["look"])
    has = E["look"].gt(0) & E["look"].lt(4)          # bậc thang / một chiều: chỉ học trên đoạn có đường mở
    has_any = E["look"].gt(0)
    ls = F.cross_entropy(oe["stairs"][has_any], E["stairs"][has_any]) if has_any.any() else 0
    lo = F.cross_entropy(oe["oneway"][has_any], E["oneway"][has_any]) if has_any.any() else 0
    on = nn_(N["x"])
    w = torch.ones(len(C.NODE_CLASSES), device=N["x"].device)
    lc = F.cross_entropy(on["content"], N["content"], weight=w)
    hm = N["hmask"]
    lh = F.cross_entropy(on["heading"][hm], N["heading"][hm]) if hm.any() else 0
    ol, ow = mn(M["leg"]), mn(M["wx"])
    lm = F.cross_entropy(ol["legend"], M["legy"]) + F.cross_entropy(ow["weather"], M["wy"])
    del has
    return le + ls + lo + lc + 2 * lh + lm


@torch.no_grad()
def patch_eval(nets, items, dev):
    for m in nets:
        m.eval()
    ds = SceneDS(items, augment=False)
    c = {k: [0, 0] for k in ["look", "stairs", "oneway", "content", "heading", "legend", "weather"]}
    for i in range(len(ds)):
        E, N, M = patch_batch([ds[i]], dev, train=False)
        oe, on = nets[0](E["x"]), nets[1](N["x"])
        ol, ow = nets[2](M["leg"]), nets[2](M["wx"])
        hasr = E["look"].gt(0)
        for k, o, y, mask in [("look", oe["look"], E["look"], None), ("stairs", oe["stairs"], E["stairs"], hasr),
                              ("oneway", oe["oneway"], E["oneway"], hasr), ("content", on["content"], N["content"], None),
                              ("heading", on["heading"], N["heading"], N["hmask"]),
                              ("legend", ol["legend"], M["legy"], None), ("weather", ow["weather"], M["wy"], None)]:
            p = o.argmax(1)
            if mask is not None:
                p, y = p[mask], y[mask]
            c[k][0] += (p == y).sum().item()
            c[k][1] += len(y)
    for m in nets:
        m.train()
    return {k: v[0] / max(v[1], 1) for k, v in c.items()}


def train_patch(a, dev):
    items, vA = load_items(a.use_val_a, a.use_val_all)
    items = oversample(items, a.big_rep)
    nets = [edge_net().to(dev), node_net().to(dev), misc_net().to(dev)]
    if a.init_from:
        for name, m in zip(["edge", "node", "misc"], nets):
            m.load_state_dict(torch.load(Path(a.init_from) / f"{name}.pt", map_location="cpu", weights_only=True))
    print("tham số: edge", n_params(nets[0]), "node", n_params(nets[1]), "misc", n_params(nets[2]))
    scaler = torch.amp.GradScaler()
    params = [p for m in nets for p in m.parameters()]
    lr = a.lr or 2e-3
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = a.epochs * (len(items) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        dl = DataLoader(SceneDS(items, True, ep), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                        collate_fn=collate, drop_last=True, generator=torch.Generator().manual_seed(SEED + ep))
        t0, tot = time.time(), 0.0
        for batch in dl:
            E, N, M = patch_batch(batch, dev, train=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = patch_losses(nets, E, N, M)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
        msg = ""
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            acc = patch_eval(nets, vA[:60], dev)
            msg = "  val-A[60]: " + " ".join(f"{k} {v * 100:.2f}" for k, v in acc.items())
        print(f"[patch] epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f} ({time.time() - t0:.0f}s){msg}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, m in zip(["edge", "node", "misc"], nets):
        torch.save(m.state_dict(), a.out / f"{name}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["kp", "patch"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--use-val-a", action="store_true", help="thêm ảnh val-A vào dữ liệu huấn luyện")
    ap.add_argument("--use-val-all", action="store_true", help="thêm TOÀN BỘ ảnh validation (bản nộp cuối)")
    ap.add_argument("--aug", type=int, default=1, help="bản tăng cường dữ liệu (1: bước 4, 2: bước 5C)")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--kp-input", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--big-rep", type=int, default=1, help="lặp ảnh lưới 9 (k lần) / lưới 8 ((k+1)//2 lần); 1 = tắt")
    ap.add_argument("--init-from", default=None, help="thư mục trọng số để học tiếp")
    ap.add_argument("--out", type=Path, default=MODELS, help="thư mục lưu (mặc định models/)")
    a = ap.parse_args()
    C.AUG_VERSION = a.aug
    C.KP_INPUT = a.kp_input
    C.seed_all(SEED)
    torch.backends.cudnn.benchmark = a.stage == "kp"   # mảnh ảnh: số lượng thay đổi -> không dò cấu hình
    dev = torch.device("cuda")
    (train_kp if a.stage == "kp" else train_patch)(a, dev)
    # thư mục mô hình phải đủ 4 mạng + cấu hình: chép mạng còn thiếu từ models/ (bản đang dùng)
    import json
    import shutil
    for name in ("kp", "edge", "node", "misc"):
        if not (a.out / f"{name}.pt").exists():
            shutil.copy(MODELS / f"{name}.pt", a.out / f"{name}.pt")
    cfg_p = a.out / "config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    if a.stage == "kp":
        cfg.update(backbone=a.backbone, kp_input=a.kp_input)
    cfg.setdefault("backbone", "resnet18")
    cfg.setdefault("kp_input", 512)
    cfg_p.write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
