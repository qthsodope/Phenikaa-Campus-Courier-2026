"""Bộ phận phụ của khối 2 đã học sẵn (bước 6.1): kho từ vựng, lưới an toàn (CharNet), mô hình n-gram cờ (FlagNet).

Khi dự đoán, nạp từ models/nlp_aux_<chế độ>.pkl — KHÔNG học lại, KHÔNG đọc dữ liệu train/validation lúc chạy.
Nếu file chưa có (lúc phát triển), các bộ phận tự học như trước. Dựng file: scripts/build_nlp_aux.py.
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_DATA = "unloaded"


def path():
    from .mode import MODE, TAG
    return ROOT / "models" / f"nlp_aux_{MODE}{TAG}.pkl"


def get(name):
    """name ∈ {"vocab", "safety", "flagnet"} -> đối tượng đã lưu, hoặc None nếu chưa có file."""
    global _DATA
    if _DATA == "unloaded":
        p = path()
        _DATA = pickle.loads(p.read_bytes()) if p.exists() else None
    return None if _DATA is None else _DATA.get(name)


def build_and_save():
    from . import text, parse, flagnet
    from .safety import build_default_net
    from .augment import Pools
    global _DATA
    _DATA = None
    voc = text.build_vocab(__import__("courier.nlp.mode", fromlist=["x"]).train_texts(), text.lexicon_phrases())
    text._VOCAB = voc
    net = build_default_net()
    parse._NET = net
    fn = flagnet.get_flagnet()
    data = {"vocab": voc, "safety": net, "flagnet": fn}
    path().write_bytes(pickle.dumps(data))
    _DATA = data
    return path()
