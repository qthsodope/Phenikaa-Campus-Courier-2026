"""Dựng và lưu các bộ phận phụ của khối 2 (kho từ vựng, lưới an toàn, n-gram cờ) -> models/nlp_aux_<chế độ>.pkl.

    COURIER_NLP_MODE=final .venv/bin/python scripts/build_nlp_aux.py     # bản nộp (train + validation V2)
    COURIER_NLP_MODE=blind .venv/bin/python scripts/build_nlp_aux.py     # chế độ mù (chỉ train V2)
"""
import os
import sys
from pathlib import Path

os.environ["COURIER_NLP_MODEL"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from courier.nlp import aux  # noqa: E402

print("đã lưu", aux.build_and_save())
