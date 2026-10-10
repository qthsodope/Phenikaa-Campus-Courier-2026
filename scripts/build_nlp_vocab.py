"""Dựng bảng "món hàng -> nơi giao" (chỉ dùng câu mission của TRAIN). Kho từ vựng giờ dựng trong bộ nhớ (text.vocab).

    .venv/bin/python scripts/build_nlp_vocab.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courier.data import load_split  # noqa: E402

# ----------------------------------------------------------------- món hàng -> nơi giao (dự phòng)
# Học từ TRAIN (chỉ các cảnh không có điểm ghé): món hàng nào thường được giao tới loại địa điểm nào.
import re  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

from courier.nlp.parse import parse_tokens  # noqa: E402
from courier.nlp.text import find_places, ITEM_HINTS_PATH  # noqa: E402

_, _, scenes = load_split("train")
PATS = [r"\bcan nhan (.+?) \.", r"\bdang cho (.+?) \.", r"\bco don giao (.+?) o\b", r"^(?:.*: )?can (.+?) o\b",
        r"\b(?:mang|gui|dua|giao|chuyen) (.+?) (?:toi|den|qua)\b", r"hang : (.+?) \."]
assoc = defaultdict(Counter)
for s in scenes:
    if s["mission"]["via"]:
        continue
    t = " ".join(parse_tokens(s["mission"]["text"]))
    for sent in re.split(r"(?<=[.!?]) ", t):
        for p in PATS:
            for m in re.finditer(p, sent):
                item = m.group(1)
                if len(item.split()) <= 4 and not find_places(item.split()):   # bỏ cụm chứa tên địa điểm
                    assoc[item][s["mission"]["goal"]] += 1
hints = {}
for item, cnt in assoc.items():
    n = sum(cnt.values())
    typ, k = cnt.most_common(1)[0]
    if n >= 3 and k / n >= 0.6:
        hints[item] = typ
ITEM_HINTS_PATH.write_text(json.dumps(hints, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8")
print(f"đã ghi {ITEM_HINTS_PATH}: {len(hints)} món hàng")
