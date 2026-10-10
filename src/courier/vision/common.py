"""Phần dùng chung cho khối đọc ảnh: hằng số, đọc ảnh, tăng cường dữ liệu, cắt mảnh ảnh, tạo nhãn."""
import io
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

PLACE_TYPES = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
LOOKS = ["normal", "crowded", "covered"]            # "kiểu nhìn" thông thường của 3 trạng thái
EDGE_CLASSES = ["none", "look_normal", "look_crowded", "look_covered", "closed"]
NODE_CLASSES = ["none"] + PLACE_TYPES + ["robot"]
HEADINGS = ["UP", "DOWN", "LEFT", "RIGHT"]
KP_CHANNELS = ["node", "leg_normal", "leg_crowded", "leg_covered", "weather"]

# kích thước mảnh ảnh (pixel ảnh gốc -> pixel đưa vào mạng)
EDGE_H, EDGE_W, EDGE_HALF_THICK = 32, 96, 18          # dải dọc đoạn nối, dày 2*18 px ảnh gốc
NODE_SIZE, NODE_REGION = 80, 124                       # ô vuông 124 px quanh giao lộ -> 80x80
MISC_SIZE, LEG_REGION, WEATHER_REGION = 64, 72, 84     # ô mẫu chú giải / ô thời tiết -> 64x64
KP_INPUT, KP_STRIDE = 512, 4                           # mạng dò điểm: ảnh thu về 512x512, bản đồ nhiệt 128x128

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_rgb(path):
    """Mở ảnh -> RGB 8-bit, chịu được mọi chế độ màu: bảng màu (P, kể cả có màu trong suốt), RGBA/LA (ghép lên nền
    trắng thay vì nền đen), ảnh xám, ảnh 16-bit, CMYK. Ảnh train/val (P, RGB) cho kết quả y như convert("RGB")."""
    im = Image.open(path)
    im.load()
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, dtype=np.float64)
        im = Image.fromarray((a / max(a.max(), 1) * 255).astype(np.uint8))
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return im.convert("RGB")


# ----------------------------------------------------------------- tăng cường dữ liệu (giống val/test)
AUG = {   # bản 1 (bước 4) và bản 2 (bước 5C: đuôi nặng hơn, phủ cả mức "làm xấu thêm" 1-2 của bài thử độ bền)
    1: dict(p_rot=0.6, rot=3.0, p_blur=0.35, blur=(0.3, 1.2), p_jpeg=0.45, q=(40, 90), p_heavy=0.0),
    2: dict(p_rot=0.7, rot=4.5, p_blur=0.45, blur=(0.3, 1.6), p_jpeg=0.5, q=(25, 90), p_heavy=0.2),
    # bản 3 (08/10 tối, sau báo cáo chênh lệch test): đuôi nặng dày hơn, xoay tới 5,5° — ảnh test có thể xấu hơn validation
    3: dict(p_rot=0.8, rot=5.5, p_blur=0.5, blur=(0.3, 1.8), p_jpeg=0.6, q=(20, 90), p_heavy=0.4),
    # bản 4 (09/10 rạng sáng): bản 3 + đổi màu vừa phải cho CẢ ảnh (chú giải đổi theo) — khối ảnh không bám đúng bảng màu
    4: dict(p_rot=0.8, rot=5.5, p_blur=0.5, blur=(0.3, 1.8), p_jpeg=0.6, q=(20, 90), p_heavy=0.4, p_color=0.7),
    # bản 5 (09/10 trưa, "bảo hiểm" ảnh test xấu hơn validation): xoay tới 7°, mờ tới 2,4, JPEG tới 15, đuôi nặng 0,5
    5: dict(p_rot=0.85, rot=7.0, p_blur=0.55, blur=(0.3, 2.4), p_jpeg=0.65, q=(15, 90), p_heavy=0.5),
}


def color_jitter(img, rng):
    """Đổi màu vừa phải cho cả ảnh: độ sáng 0,7-1,3, tương phản 0,7-1,3, bão hòa 0,6-1,4, sắc độ lệch +-12 độ."""
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))
    d = int(round(rng.uniform(-12, 12) / 360 * 255))
    if d:
        h, s_, v = img.convert("HSV").split()
        img = Image.merge("HSV", (h.point(lambda x: (x + d) % 256), s_, v)).convert("RGB")
    return img
AUG_VERSION = 1


def augment(img, pts, rng, strength=1.0):
    """Xoay nhẹ / mờ / nén JPEG ngẫu nhiên. pts: mảng (N,2) tọa độ pixel, được biến đổi theo phép xoay.
    Bản 2: thêm 20% ca "nặng" (mờ 1,2-2,4, JPEG 12-40) để chịu được ảnh xấu hơn mức thường gặp."""
    A = AUG[AUG_VERSION]
    W, H = img.size
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    heavy = rng.random() < A["p_heavy"]
    if rng.random() < A.get("p_color", 0.0):
        img = color_jitter(img, rng)
    if rng.random() < A["p_rot"] * strength:
        ang = rng.uniform(-A["rot"], A["rot"])
        bg = img.getpixel((2, 2))
        img = img.rotate(ang, resample=Image.BICUBIC, center=(W / 2, H / 2), fillcolor=bg)
        t = math.radians(ang)       # PIL xoay ngược chiều kim đồng hồ (trục y hướng xuống)
        c, s = math.cos(t), math.sin(t)
        x, y = pts[:, 0] - W / 2, pts[:, 1] - H / 2
        pts = np.stack([c * x + s * y + W / 2, -s * x + c * y + H / 2], 1)
    if heavy or rng.random() < A["p_blur"] * strength:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.2, 2.4) if heavy else rng.uniform(*A["blur"])))
    if heavy or rng.random() < A["p_jpeg"] * strength:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(12, 40) if heavy else rng.randint(*A["q"]))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img, pts


def degrade(img, pts, rng, level):
    """Làm xấu ảnh MẠNH HƠN thường gặp (thử độ bền, B4). level 1: mờ 0,8-1,5, JPEG 30-55, xoay 2-4°;
    level 2: mờ 1,3-2,2, JPEG 15-35, xoay 3-5°. Luôn áp đủ ba phép. pts biến đổi theo phép xoay."""
    W, H = img.size
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    lo_b, hi_b, lo_q, hi_q, lo_r, hi_r = ((0.8, 1.5, 30, 55, 2.0, 4.0) if level == 1 else (1.3, 2.2, 15, 35, 3.0, 5.0))
    ang = rng.uniform(lo_r, hi_r) * rng.choice([-1, 1])
    bg = img.getpixel((2, 2))
    img = img.rotate(ang, resample=Image.BICUBIC, center=(W / 2, H / 2), fillcolor=bg)
    t = math.radians(ang)
    c, s = math.cos(t), math.sin(t)
    x, y = pts[:, 0] - W / 2, pts[:, 1] - H / 2
    pts = np.stack([c * x + s * y + W / 2, -s * x + c * y + H / 2], 1)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(lo_b, hi_b)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(lo_q, hi_q))
    buf.seek(0)
    return Image.open(buf).convert("RGB"), pts


def to_tensor(img):
    """PIL -> tensor float (3,H,W) đã chuẩn hóa."""
    a = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return (a - MEAN) / STD


# ----------------------------------------------------------------- cắt mảnh ảnh bằng grid_sample
def _sample(img_t, centers, ux, uy, half_w, half_h, out_h, out_w):
    """Cắt các mảnh hình chữ nhật xoay. img_t: (3,H,W) trên device. centers (N,2) pixel; ux,uy (N,2) vectơ đơn vị
    theo chiều rộng / chiều cao của mảnh; half_w, half_h (N,) nửa kích thước theo pixel ảnh gốc."""
    _, H, W = img_t.shape
    dev = img_t.device
    n = centers.shape[0]
    xs = torch.linspace(-1, 1, out_w, device=dev)
    ys = torch.linspace(-1, 1, out_h, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")                      # (h,w)
    gx = gx[None] * half_w[:, None, None]
    gy = gy[None] * half_h[:, None, None]
    px = centers[:, 0, None, None] + gx * ux[:, 0, None, None] + gy * uy[:, 0, None, None]
    py = centers[:, 1, None, None] + gx * ux[:, 1, None, None] + gy * uy[:, 1, None, None]
    grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)  # (N,h,w,2)
    return F.grid_sample(img_t[None].expand(n, -1, -1, -1), grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def edge_patches(img_t, a, b):
    """Dải ảnh dọc đoạn a->b (a, b: tensor (N,2)). Chiều rộng mảnh = hướng a->b."""
    d = b - a
    L = d.norm(dim=1).clamp(min=1.0)
    ux = d / L[:, None]
    uy = torch.stack([-ux[:, 1], ux[:, 0]], 1)
    half_w = L / 2
    half_h = torch.full_like(L, EDGE_HALF_THICK)
    return _sample(img_t, (a + b) / 2, ux, uy, half_w, half_h, EDGE_H, EDGE_W)


def square_patches(img_t, centers, region, size):
    n = centers.shape[0]
    dev = img_t.device
    ux = torch.tensor([[1.0, 0.0]], device=dev).expand(n, 2)
    uy = torch.tensor([[0.0, 1.0]], device=dev).expand(n, 2)
    half = torch.full((n,), region / 2, device=dev)
    return _sample(img_t, centers, ux, uy, half, half, size, size)


# ----------------------------------------------------------------- nhãn từ scenes.json
def scene_targets(s):
    """Trích nhãn của một cảnh: danh sách điểm (để biến đổi khi xoay) + nhãn cho từng loại mảnh."""
    nodes = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    inv = {}                               # trạng thái -> kiểu nhìn
    for st, look in s["road_look"].items():
        inv[st] = look
    edges = {}
    for e in s["edges"]:
        a, b = tuple(e["a"]), tuple(e["b"])
        if b < a:
            a, b = b, a
        if e["status"] == "closed":
            cls = 4
        else:
            cls = 1 + LOOKS.index(inv[e["status"]])
        ow = 0
        if e["oneway_to"] is not None:
            ow = 1 if tuple(e["oneway_to"]) == b else 2
        edges[(a, b)] = (cls, int(e["stairs"]), ow)
    pairs = []                             # mọi cặp giao lộ kề nhau (cả có và không có đường)
    for (r, c) in nodes:
        for nb in ((r, c + 1), (r + 1, c)):
            if nb in nodes:
                pairs.append(((r, c), nb, edges.get(((r, c), nb), (0, 0, 0))))
    content = {rc: 0 for rc in nodes}
    for lm in s["landmarks"]:
        content[tuple(lm["rc"])] = 1 + PLACE_TYPES.index(lm["type"])
    content[tuple(s["robot"]["rc"])] = len(NODE_CLASSES) - 1
    leg = {l["kind"]: l["swatch"] for l in s["legend"]}
    legend = []
    for kind in LOOKS:                     # ô mẫu của trạng thái `kind` được vẽ theo kiểu nhìn road_look[kind]
        x0, y0, x1, y1 = leg[kind]
        legend.append(((x0 + x1) / 2, (y0 + y1) / 2, LOOKS.index(s["road_look"][kind])))
    wb = s["weather_box"]
    return {
        "node_rc": list(nodes), "node_xy": [nodes[k] for k in nodes], "content": [content[k] for k in nodes],
        "heading": HEADINGS.index(s["robot"]["heading"]), "pairs": pairs, "legend": legend,
        "weather_xy": ((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2), "rain": int(s["weather"] == "rain"),
        "grid": (s["grid"]["rows"], s["grid"]["cols"]),
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
