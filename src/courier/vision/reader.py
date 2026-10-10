"""Đọc một ảnh bản đồ -> thông tin bản đồ (giống scenes.json) -> Scene cho khối 3.

Các bước:
  1. Mạng dò điểm (KeypointNet) trên ảnh thu nhỏ 512: tâm giao lộ, 3 ô mẫu chú giải (thường/đông/mái che),
     ô thời tiết.
  2. Dựng lưới: ước lượng góc xoay, gom tâm giao lộ thành hàng / cột -> (hàng, cột) cho từng giao lộ.
  3. Mỗi cặp giao lộ kề nhau: cắt dải ảnh dọc đoạn nối -> EdgeNet: không có đường / kiểu nhìn / đóng,
     bậc thang, một chiều + chiều.
  4. Mỗi giao lộ: cắt ô vuông -> NodeNet: trống / loại địa điểm / robot (+ hướng mũi).
  5. Ô mẫu chú giải -> MiscNet: mỗi ô vẽ theo kiểu nhìn nào -> cách hoán đổi (road_look) của CHÍNH ảnh này.
     Ô thời tiết -> MiscNet: mưa / khô.
  6. Đổi "kiểu nhìn" của từng đoạn sang trạng thái thật bằng road_look.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import common as C
from .models import KeypointNet, edge_net, node_net, misc_net, n_params
from ..graph import Scene, Edge, Mission, HEADING

MODELS = Path(__file__).resolve().parents[3] / "models"
STAIRS_T = float(__import__("os").environ.get("COURIER_STAIRS_T", "0.5"))   # bậc thang nếu P(bậc) > ngưỡng (0,5 = argmax như BEST). Thử 0,1: chỉnh trên ảnh train làm xấu (mức 1: sót 9->4, mức 2: 192->151; thừa 0)
ONEWAY_T = None    # None = argmax; số t: một chiều nếu P(không một chiều) < t (hướng = lớp một chiều lớn hơn)
NODE_THRESHOLD = 0.05      # ngưỡng bản đồ nhiệt cho tâm giao lộ (chọn trên val-A)
# bước 5C: đỉnh yếu (LOW..NODE_THRESHOLD) được hậu kiểm bằng cấu trúc lưới; None = tắt. Đặt bằng COURIER_LOW_THR.
ROT_TTA = int(__import__("os").environ.get("COURIER_ROT_TTA", "0"))   # H25: đọc xoay 90/180/270 + biểu quyết
LM_MAX2 = int(__import__("os").environ.get("COURIER_LM_MAX2", "1"))   # 10/10: ép "mỗi loại <= 2 bản" (đề mục 3)
LOW_THRESHOLD = (lambda v: None if v in ("", "none") else float(v))(__import__("os").environ.get("COURIER_LOW_THR", "0.02"))


PREPROC = __import__("os").environ.get("COURIER_PREPROC", "")   # H29: median | bilateral | nlm3 | nlm7 | clahe (+ "_noisy")
NOISY_T = float(__import__("os").environ.get("COURIER_NOISY_T", "0"))


def _is_night(img):
    return bool(np.median(np.asarray(img.convert("L").resize((128, 96)), dtype=np.float32)) < 128)


def preprocess(img):
    """H29: làm sạch ảnh trước khi đọc (OpenCV). '_noisy': chỉ áp dụng khi độ lệch chuẩn Laplacian > NOISY_T.
    clahe: chỉ ảnh ngày (ảnh đêm giữ nguyên)."""
    import cv2
    from PIL import Image
    kind = PREPROC.replace("_noisy", "")
    a = np.asarray(img.convert("RGB"))
    if PREPROC.endswith("_noisy"):
        if cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).std() <= NOISY_T:
            return img
    if kind == "median":
        a = cv2.medianBlur(a, 3)
    elif kind == "bilateral":
        a = cv2.bilateralFilter(a, 5, 40, 40)
    elif kind.startswith("nlm"):
        h = int(kind[3:] or 5)
        a = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(a), None, h, h, 7, 21)
    elif kind == "clahe":
        if _is_night(img):
            return img
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        a = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(a)


class _Avg(torch.nn.Module):
    """Ghép mạng: trung bình xác suất, trả lại dạng 'logit' để phần giải mã giữ nguyên (sigmoid / softmax của đầu ra
    = trung bình xác suất của các mạng)."""
    def __init__(self, ms):
        super().__init__()
        self.ms = torch.nn.ModuleList(ms)

    def forward(self, x):
        outs = [m(x) for m in self.ms]
        if isinstance(outs[0], dict):
            return {k: torch.log(torch.stack([F.softmax(o[k].float(), 1) for o in outs]).mean(0).clamp_min(1e-9))
                    for k in outs[0]}
        p = torch.stack([torch.sigmoid(o.float()) for o in outs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)


class VisionReader:
    def __init__(self, models_dir=MODELS, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg_path = Path(models_dir) / "config.json"                  # bản 2: nền + độ phân giải đầu vào
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.kp_input = cfg.get("kp_input", C.KP_INPUT)
        self.node_threshold = cfg.get("node_threshold", NODE_THRESHOLD)
        self.low_threshold = cfg.get("low_threshold", LOW_THRESHOLD)
        self.tta_shift = cfg.get("tta_shift", float(__import__("os").environ.get("COURIER_TTA", "0")))
        self.kp = KeypointNet(pretrained=False, backbone=cfg.get("backbone", "resnet18"))
        self.edge, self.node, self.misc = edge_net(), node_net(), misc_net()
        for name, m in [("kp", self.kp), ("edge", self.edge), ("node", self.node), ("misc", self.misc)]:
            m.load_state_dict(torch.load(Path(models_dir) / f"{name}.pt", map_location="cpu", weights_only=True))
            m.to(self.dev).eval()
        # H26 (10/10): ghép nhiều bộ trọng số — trung bình XÁC SUẤT (sigmoid bản đồ nhiệt / softmax mảnh ảnh) rồi giải mã
        extra = cfg.get("ensemble", []) + [d for d in __import__("os").environ.get("COURIER_VIS_ENSEMBLE", "").split(",") if d]
        for d in extra:
            d = Path(d) if Path(d).is_absolute() or Path(d).exists() else Path(models_dir) / d
            others = {"kp": KeypointNet(pretrained=False, backbone=cfg.get("backbone", "resnet18")), "edge": edge_net(),
                      "node": node_net(), "misc": misc_net()}
            for name, m in others.items():
                m.load_state_dict(torch.load(d / f"{name}.pt", map_location="cpu", weights_only=True))
                m.to(self.dev).eval()
                cur = getattr(self, name)
                setattr(self, name, _Avg((cur.ms if isinstance(cur, _Avg) else [cur]) + [m]))

    def n_params(self):
        return {k: n_params(m) for k, m in [("kp", self.kp), ("edge", self.edge), ("node", self.node),
                                            ("misc", self.misc)]}

    # ------------------------------------------------------------- 1. dò điểm
    @torch.no_grad()
    def keypoints(self, im):
        _, H, W = im.shape
        S = self.kp_input
        sc = S / max(H, W)
        x = torch.zeros(1, 3, S, S, device=self.dev)
        r = F.interpolate(im[None], size=(round(H * sc), round(W * sc)), mode="bilinear", align_corners=False)
        x[0, :, :r.shape[2], :r.shape[3]] = r[0]
        hm = torch.sigmoid(self.kp(x)[0].float())                      # (5,R,R)

        def to_img(cy, cx):
            return ((cx + 0.5) * C.KP_STRIDE / sc, (cy + 0.5) * C.KP_STRIDE / sc)

        def refine(ch, y, x_):
            y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, hm.shape[1]), max(x_ - 1, 0), min(x_ + 2, hm.shape[2])
            w = hm[ch, y0:y1, x0:x1]
            yy, xx = torch.meshgrid(torch.arange(y0, y1, device=self.dev), torch.arange(x0, x1, device=self.dev),
                                    indexing="ij")
            s = w.sum()
            return float((w * yy).sum() / s), float((w * xx).sum() / s)

        peak = (hm == F.max_pool2d(hm[None], 3, 1, 1)[0])
        lo = self.node_threshold if self.low_threshold is None else min(self.low_threshold, self.node_threshold)
        ys, xs = torch.nonzero(peak[0] & (hm[0] > lo), as_tuple=True)
        nodes = []
        for y, x_ in zip(ys.tolist(), xs.tolist()):
            cy, cx = refine(0, y, x_)
            nodes.append((*to_img(cy, cx), float(hm[0, y, x_])))
        singles = []
        for ch in range(1, 5):
            k = int(torch.argmax(hm[ch]))
            y, x_ = divmod(k, hm.shape[2])
            cy, cx = refine(ch, y, x_)
            singles.append((*to_img(cy, cx), float(hm[ch, y, x_])))
        return nodes, singles

    # ------------------------------------------------------------- 2. dựng lưới
    @staticmethod
    def build_grid(nodes):
        P = np.array([(x, y) for x, y, _ in nodes], dtype=np.float64)
        score = np.array([s for _, _, s in nodes])
        if len(P) < 4:
            return None
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        np.fill_diagonal(D, np.inf)
        nn_d = D.min(1)
        spacing = float(np.median(nn_d))

        def cluster(v):
            order = np.argsort(v)
            lab = np.zeros(len(v), dtype=int)
            k = 0
            for a, b in zip(order[:-1], order[1:]):
                if v[b] - v[a] > 0.45 * spacing:
                    k += 1
                lab[b] = k
            return lab, k + 1

        def spread(v):                      # tổng độ trải trong từng nhóm + phạt mỗi nhóm thêm (tránh tách đôi hàng)
            lab, n = cluster(v)
            return sum(np.ptp(v[lab == k]) for k in range(n)) + 0.5 * spacing * n

        def rotate(deg):
            t = math.radians(deg)
            c, s = math.cos(t), math.sin(t)
            Z = P - P.mean(0)
            return np.stack([c * Z[:, 0] + s * Z[:, 1], -s * Z[:, 0] + c * Z[:, 1]], 1)

        # góc xoay: thử -4..4 độ, chọn góc làm các hàng + cột "gọn" nhất (tổng độ trải trong mỗi hàng/cột nhỏ nhất)
        cands = np.arange(-4.0, 4.0001, 0.1)
        costs = [spread(rotate(d)[:, 1]) + spread(rotate(d)[:, 0]) for d in cands]
        deg = float(cands[int(np.argmin(costs))])
        Q = rotate(deg)
        th = math.radians(deg)

        rlab, nr = cluster(Q[:, 1])
        clab, nc = cluster(Q[:, 0])
        best = {}
        for i in range(len(P)):
            rc = (int(rlab[i]), int(clab[i]))
            if rc not in best or score[i] > score[best[rc]]:
                best[rc] = i
        row_c = [float(np.median(Q[rlab == k, 1])) for k in range(nr)]
        col_c = [float(np.median(Q[clab == k, 0])) for k in range(nc)]
        return {"rows": nr, "cols": nc, "xy": {rc: tuple(P[i]) for rc, i in best.items()}, "angle": math.degrees(th),
                "deg": deg, "mean": P.mean(0), "row_c": row_c, "col_c": col_c, "spacing": spacing}

    @staticmethod
    def recover_weak(g, weak, tol=0.3):
        """Hậu kiểm bằng cấu trúc lưới (bước 5C): điểm YẾU (đỉnh nhiệt dưới ngưỡng chính) chỉ được nhận khi nằm đúng
        giao của một hàng và một cột đã dựng từ các điểm chắc chắn (sai lệch < tol x khoảng cách giao lộ) và ô đó
        đang trống. Điểm thừa (góc ảnh, ký hiệu trên đường, chữ) thường không thỏa cả hai điều kiện."""
        t = math.radians(g["deg"])
        c, s_ = math.cos(t), math.sin(t)
        occupied = set(g["xy"])
        out = []
        for x, y, sc in sorted(weak, key=lambda n: -n[2]):
            zx, zy = x - g["mean"][0], y - g["mean"][1]
            qx, qy = c * zx + s_ * zy, -s_ * zx + c * zy
            r = int(np.argmin([abs(qy - v) for v in g["row_c"]]))
            k = int(np.argmin([abs(qx - v) for v in g["col_c"]]))
            if (abs(qy - g["row_c"][r]) < tol * g["spacing"] and abs(qx - g["col_c"][k]) < tol * g["spacing"]
                    and (r, k) not in occupied):
                occupied.add((r, k))
                out.append((x, y, sc))
        return out

    def _tta(self, net, crop, n):
        """Suy luận nhiều lần rồi gộp (bước 5C): cắt mảnh tại chỗ + lệch nhẹ TTA_SHIFT pixel theo 4 hướng chéo,
        lấy TRUNG BÌNH xác suất. TTA_SHIFT = 0 -> chỉ cắt một lần như cũ."""
        offs = [(0.0, 0.0)] + ([(d, d), (-d, -d), (d, -d), (-d, d)] if (d := self.tta_shift) else [])
        acc = None
        for ox, oy in offs:
            o = torch.tensor([[ox, oy]], dtype=torch.float32, device=self.dev)
            out = {k: F.softmax(v.float(), 1) for k, v in net(crop(o)).items()}
            acc = out if acc is None else {k: acc[k] + out[k] for k in acc}
        return {k: v / len(offs) for k, v in acc.items()}

    # ------------------------------------------------------------- toàn bộ
    def read(self, path):
        """ROT_TTA=1 (H25): đọc thêm ảnh xoay 90/180/270 rồi biểu quyết (vision/rot_tta.py)."""
        if PREPROC:
            path = preprocess(path if hasattr(path, "convert") else C.load_rgb(path))
        if not ROT_TTA:
            return self._read1(path)
        from . import rot_tta
        img = path if hasattr(path, "convert") else C.load_rgb(path)
        reads = [self._read1(img)]
        for k in (1, 2, 3):
            try:
                v = self._read1(img.rotate(90 * k, expand=True))
                reads.append(rot_tta.unrotate(v, k) if v is not None else None)
            except Exception:
                reads.append(None)
        return rot_tta.vote(reads)

    @torch.no_grad()
    def _read1(self, path):
        img = path if hasattr(path, "convert") else C.load_rgb(path)     # nhận đường dẫn hoặc ảnh PIL
        im = C.to_tensor(img).to(self.dev)
        nodes, singles = self.keypoints(im)
        strong = [n for n in nodes if n[2] > self.node_threshold]
        g = self.build_grid(strong)
        if g is None:
            return None
        if self.low_threshold is not None:            # hậu kiểm bằng cấu trúc lưới: cứu giao lộ có đỉnh nhiệt yếu
            extra = self.recover_weak(g, [n for n in nodes if n[2] <= self.node_threshold])
            if extra:
                g = self.build_grid(strong + extra)
                nodes = strong + extra
            else:
                nodes = strong
        else:
            nodes = strong
        xy = g["xy"]
        rcs = sorted(xy)
        T = torch.tensor([xy[rc] for rc in rcs], dtype=torch.float32, device=self.dev)
        idx = {rc: k for k, rc in enumerate(rcs)}

        # 3. đoạn đường
        pairs = [(rc, nb) for rc in rcs for nb in ((rc[0], rc[1] + 1), (rc[0] + 1, rc[1])) if nb in idx]
        edges = []
        if pairs:
            a = T[[idx[p[0]] for p in pairs]]
            b = T[[idx[p[1]] for p in pairs]]
            oe = self._tta(self.edge, lambda o: C.edge_patches(im, a + o, b + o), len(pairs))
            look = oe["look"].cpu().numpy()
            ps = oe["stairs"].cpu().numpy()
            po = oe["oneway"].cpu().numpy()
            # ngưỡng (mặc định = argmax). Ảnh mờ/nén nặng làm mô hình lệch về "không có ký hiệu" -> có thể hạ ngưỡng
            stairs = ps[:, 1] > STAIRS_T
            ow = po.argmax(1) if ONEWAY_T is None else np.where(po[:, 0] < ONEWAY_T, 1 + po[:, 1:].argmax(1), 0)
            for k, (p, q) in enumerate(pairs):
                edges.append({"a": p, "b": q, "look": int(look[k].argmax()), "p": look[k],
                              "stairs": bool(stairs[k]), "oneway": int(ow[k]), "ps": float(ps[k, 1]), "po": po[k]})

        # 4. giao lộ
        on = self._tta(self.node, lambda o: C.square_patches(im, T + o, C.NODE_REGION, C.NODE_SIZE), len(rcs))
        pc = on["content"].cpu().numpy()
        ph = on["heading"].cpu().numpy()
        ROBOT = len(C.NODE_CLASSES) - 1
        r_idx = int(pc[:, ROBOT].argmax())
        landmarks = {}
        pcl = pc[:, :ROBOT].copy()
        if LM_MAX2:        # đề: mỗi loại địa điểm có tối đa HAI bản -> loại thừa: bỏ bản kém chắc, giao lộ đó lấy lựa chọn kế
            while True:
                cls_ = pcl.argmax(1)
                over = [t for t in range(1, ROBOT) if sum(1 for k in range(len(rcs)) if k != r_idx and cls_[k] == t) > 2]
                if not over:
                    break
                for t in over:
                    ks = sorted((k for k in range(len(rcs)) if k != r_idx and cls_[k] == t), key=lambda k: (-pcl[k, t], k))
                    for k in ks[2:]:
                        pcl[k, t] = -1.0
        for k, rc in enumerate(rcs):
            if k == r_idx:
                continue
            cls = int(pcl[k].argmax())
            if cls > 0:
                landmarks.setdefault(C.PLACE_TYPES[cls - 1], []).append(rc)

        # 5. chú giải + thời tiết
        leg = torch.tensor([s_[:2] for s_ in singles[:3]], dtype=torch.float32, device=self.dev)
        pl = F.softmax(self.misc(C.square_patches(im, leg, C.LEG_REGION, C.MISC_SIZE))["legend"].float(), 1)
        pl = pl.cpu().numpy()                                          # pl[kind, look]
        best, road_look = -1, None
        for perm in itertools.permutations(range(3)):
            v = sum(math.log(pl[k, perm[k]] + 1e-9) for k in range(3))
            if v > best or road_look is None:
                best, road_look = v, {C.LOOKS[k]: C.LOOKS[perm[k]] for k in range(3)}
        wc = torch.tensor([singles[3][:2]], dtype=torch.float32, device=self.dev)
        pw = F.softmax(self.misc(C.square_patches(im, wc, C.WEATHER_REGION, C.MISC_SIZE))["weather"].float(), 1)
        rain = bool(pw[0, 1] > 0.5)
        # ảnh "đêm": nền tối. Độ sáng trung vị: đêm 32-35, ngày 233-251 (train + validation) -> ngưỡng 128
        night = bool(np.median(np.asarray(img.convert("L").resize((128, 96)), dtype=np.float32)) < 128)

        # 6. kiểu nhìn -> trạng thái thật
        look2status = {look: st for st, look in road_look.items()}
        out_edges = []
        for e in edges:
            if e["look"] == 0:
                continue
            status = "closed" if e["look"] == 4 else look2status[C.LOOKS[e["look"] - 1]]
            oneway_to = None if e["oneway"] == 0 else (e["b"] if e["oneway"] == 1 else e["a"])
            out_edges.append({"a": e["a"], "b": e["b"], "status": status, "stairs": e["stairs"],
                              "oneway_to": oneway_to, "ps": e["ps"], "po": e["po"]})
        # H27: xác suất cho bước sửa bản đồ theo luật đề (vision/repair.py)
        pairs_p = [{"a": e["a"], "b": e["b"], "pl": [float(x) for x in e["p"]], "ps": e["ps"],
                    "po": [float(x) for x in e["po"]]} for e in edges]
        node_p = {rc: [float(x) for x in pcl[k]] for k, rc in enumerate(rcs) if k != r_idx}
        return {
            "grid": (g["rows"], g["cols"]), "nodes": rcs, "edges": out_edges, "landmarks": landmarks,
            "robot": rcs[r_idx], "heading": C.HEADINGS[int(ph[r_idx].argmax())], "rain": rain, "night": night,
            "road_look": road_look, "angle": g["angle"], "legend_xy": [tuple(map(float, s_[:2])) for s_ in singles[:3]],
            "xy": {rc: xy[rc] for rc in rcs}, "raw_nodes": nodes,     # để phân tích lỗi (không dùng khi dự đoán)
            "pairs_p": pairs_p, "node_p": node_p, "look2status": look2status,
        }


def scene_from_vision(v, mission=None):
    """Kết quả read() -> Scene (đúng loại khối 3 dùng). mission: Mission hoặc None (gán sau)."""
    edges = {frozenset((e["a"], e["b"])): Edge(e["status"], e["stairs"], e["oneway_to"]) for e in v["edges"]}
    return Scene(rows=v["grid"][0], cols=v["grid"][1], nodes=set(v["nodes"]), edges=edges,
                 landmarks={k: list(x) for k, x in v["landmarks"].items()}, robot=tuple(v["robot"]),
                 heading=HEADING[v["heading"]], rain=v["rain"], mission=mission or Mission(goal=None),
                 night=v.get("night", False))
