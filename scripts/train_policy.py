"""Khối 3 V2: học mô hình "chọn bước" cho robot 1-9 (robot 0 dùng luật tường minh), trên bản đồ + yêu cầu CHUẨN.

Hai tầng: (1) bảng giá có điều kiện cho robot 1-8 (policy/cond.py); (2) cây tăng cường chọn bước, dùng đặc trưng của
dàn bộ tìm đường + của (1). Đặc trưng (1) trên dữ liệu học được tính kiểu HỌC CHÉO 5 phần (bảng giá học trên 4 phần,
tính cho phần còn lại) để cây không học "đặc trưng đã nhìn thấy nhãn".

    .venv/bin/python scripts/train_policy.py                # học trên train, đo val-A (val-B chỉ số tổng)
    .venv/bin/python scripts/train_policy.py --final        # bản nộp: học trên train + toàn bộ validation
Lưu: models/policy_v2.pkl
"""
import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courier.data import load_split, labels_by_scene  # noqa: E402
from courier.graph import scene_from_json  # noqa: E402
from courier.policy import cond  # noqa: E402
from courier.policy.meta import MetaPolicy, build_rows, with_cond  # noqa: E402
from courier.policy.rules_v2 import robot0_move  # noqa: E402
from courier.splits import val_halves  # noqa: E402

CACHE = ROOT / "cache"


def _rows(s):
    S = scene_from_json(s)
    mv, X = build_rows(S)
    return mv, X, robot0_move(S), S.heading, cond.scene_paths(S)


def rows_for(split):
    """Đặc trưng của mọi cảnh (lưu đệm theo split; xóa cache/ khi đổi BANK)."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"policy_rows3_{split}_x{cond.EXTRA2}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    _, _, sc = load_split(split)
    with mp.get_context("fork").Pool(14) as pool:
        out = pool.map(_rows, sc, chunksize=8)
    p.write_bytes(pickle.dumps(out))
    return out


# 1-8: bảng giá cộng theo điều kiện; 9: tham lam có điều kiện; 11-18: bảng giá do mạng nhỏ (robot r-10)
COND_ROBOTS = list(range(1, 10)) + [11, 13, 14, 15, 16, 17, 18]   # R2: mạng nhỏ kém hơn (kiểm chéo train) -> bỏ
COND_ROBOTS += [int(x) for x in __import__("os").environ.get("COURIER_COND_EXTRA", "").split(",") if x]   # thử nghiệm
if __import__("os").environ.get("COURIER_COND_SEED2", "0") == "1":
    COND_ROBOTS += list(range(101, 109))
FOLDS = 5


def _fit_job(args):
    r, data = args
    if r == 9:
        return cond.fit_greedy(data)
    if r > 100:                      # thử nghiệm: bảng giá thứ hai (hạt giống + điều chuẩn khác) của robot r - 100
        kw = dict(seed=2027, l1=5e-4)
        return cond.fit_dij(data, legged=(r % 10 == 4), **kw) if cond.DIJ else cond.fit(data, **kw)
    if cond.DIJ:
        return cond.fit_dij(data, legged=(r % 10 == 4), **(dict(hidden=32, l1=3e-3) if r > 10 else {}))
    return cond.fit(data, hidden=32, l1=3e-3) if r > 10 else cond.fit(data)


def cond_weights(rows, scenes, labels, idx_sets):
    """Học bảng giá có điều kiện của robot 1-9 trên từng tập chỉ số (song song). -> list {r: W} theo idx_sets."""
    jobs = []
    for idx in idx_sets:
        for r in COND_ROBOTS:
            if r == 9:
                jobs.append((r, [(*rows[i][4][2], cond.cvec(scenes[i]), labels[i][r]) for i in idx]))
            else:
                jobs.append((r, [(rows[i][4][1] if r % 10 == 4 else rows[i][4][0], cond.cvec(scenes[i]), labels[i][r % 10])
                                 + ((scenes[i],) if cond.DIJ else ()) for i in idx]))
    with mp.get_context("fork").Pool(int(__import__("os").environ.get("COURIER_PROCS", "10"))) as pool:
        Ws = pool.map(_fit_job, jobs, chunksize=1)
    out, k = [], 0
    for _ in idx_sets:
        out.append({r: Ws[k + j] for j, r in enumerate(COND_ROBOTS)})
        k += len(COND_ROBOTS)
    return out


def add_cond(rows, scenes, Ws_per_scene):
    return [(mv, with_cond(S, mv, X, paths, W), r0, h) for (mv, X, r0, h, paths), S, W in zip(rows, scenes, Ws_per_scene)]


def evaluate(name, rows, L, pol, quiet=False):
    ok = [0] * 10
    for (mv, X, r0, h), lab in zip(rows, L):
        ok[0] += r0 == lab[0]
        for r in range(1, 10):
            ok[r] += pol.predict(mv, X, r, h) == lab[r]
    n = len(rows)
    acc = [x / n * 100 for x in ok]
    w = min(range(10), key=lambda r: (acc[r], r))
    print(f"== {name} ({n} cảnh): MACRO {sum(acc) / 10:.2f}% | yếu nhất R{w} {acc[w]:.2f}")
    if not quiet:
        print("   " + " ".join(f"R{r} {a:.2f}" for r, a in enumerate(acc)))
    return sum(acc) / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--holdout-b", action="store_true", help="như --final nhưng chỉ thêm val-A; chấm val-B (số tổng)")
    ap.add_argument("--holdout-a", action="store_true", help="(thử nghiệm) thêm val-B, chấm val-A (số tổng)")
    ap.add_argument("--val-weight", type=int, default=1, help="lặp cảnh validation k lần khi học (hồ sơ 'khó' giống test)")
    a = ap.parse_args()
    t0 = time.time()
    tr = rows_for("train")
    va = rows_for("validation")
    _, lt, st = load_split("train")
    _, lv, sv = load_split("validation")
    Lt, Lv = labels_by_scene(lt), labels_by_scene(lv)
    A, B = val_halves(len(va))
    print(f"đặc trưng xong ({time.time() - t0:.0f} giây)")
    # tập học: train (+ toàn bộ validation nếu --final)
    rows, scen, labs = list(tr), list(st), list(Lt)
    K = a.val_weight
    I = list(range(len(va))) if a.final else (A if a.holdout_b else B) if (a.holdout_b or a.holdout_a) else []
    rows, scen, labs = rows + [va[i] for i in I] * K, scen + [sv[i] for i in I] * K, labs + [Lv[i] for i in I] * K
    # mã cảnh (bản lặp của cùng một cảnh validation chung mã -> cùng phần khi học chéo, không rò nhãn)
    sid = list(range(len(tr))) + [len(tr) + j for _ in range(K) for j in range(len(I))]
    S_all = [scene_from_json(s) for s in scen]
    n = len(rows)
    folds = [[i for i in range(n) if sid[i] % FOLDS != f] for f in range(FOLDS)]
    W_folds = cond_weights(rows, S_all, labs, folds + [list(range(n))])
    W_full = W_folds[-1]
    print(f"bảng giá có điều kiện xong ({time.time() - t0:.0f} giây)")
    learn = add_cond(rows, S_all, [W_folds[sid[i] % FOLDS] for i in range(n)])
    # 10/10 thử nghiệm: kiểu mô hình từng robot, vd COURIER_META_MODES="6:bag,4:pair" (mặc định: 1 cây mọi robot)
    modes = {int(k): v for k, v in (x.split(":") for x in __import__("os").environ.get("COURIER_META_MODES", "").split(",") if x)}
    pol = MetaPolicy(Ws=W_full, modes=modes).fit([(mv, X, lab) for (mv, X, _, _), lab in zip(learn, labs)])
    evaluate("train (đã học)", learn[:len(tr)], Lt, pol)
    if a.holdout_b or a.holdout_a:
        vrows = add_cond(va, [scene_from_json(s) for s in sv], [W_full] * len(va))
        I = B if a.holdout_b else A
        evaluate(f"{'val-B' if a.holdout_b else 'val-A'} (chưa học, chỉ số tổng) [val x{K}]", [vrows[i] for i in I], [Lv[i] for i in I], pol, quiet=True)
    elif not a.final:
        vrows = add_cond(va, [scene_from_json(s) for s in sv], [W_full] * len(va))
        evaluate("val-A", [vrows[i] for i in A], [Lv[i] for i in A], pol)
        evaluate("val-B (giữ kín, chỉ số tổng)", [vrows[i] for i in B], [Lv[i] for i in B], pol, quiet=True)
    if cond.DIJ:                     # (không gắn khi tắt: file giống hệt từng byte bản cũ)
        pol.dij = cond.DIJ
    if cond.R9RULE:
        pol.r9rule = cond.R9RULE
    if cond.XTAB:
        pol.xtab = cond.XTAB
    (ROOT / "models").mkdir(exist_ok=True)
    if cond.EXT:
        pol.ext = cond.EXT
    out = Path(__import__("os").environ["COURIER_POLICY_OUT"]) if __import__("os").environ.get("COURIER_POLICY_OUT") else \
        ROOT / "models" / ("policy_v2_hoB.pkl" if a.holdout_b else "policy_v2_hoA.pkl" if a.holdout_a else "policy_v2.pkl")
    out.write_bytes(pickle.dumps(pol))
    print(f"đã lưu {out} ({time.time() - t0:.0f} giây)")


if __name__ == "__main__":
    main()
