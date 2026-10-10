"""Chia validation cố định: nửa A (soi lỗi, sửa) và nửa B (giữ kín, chỉ chấm)."""
import random

SEED = 2026


def val_halves(n_scenes=300, seed=SEED):
    idx = list(range(n_scenes))
    random.Random(seed).shuffle(idx)
    half = n_scenes // 2
    return sorted(idx[:half]), sorted(idx[half:])   # (A, B)
