"""Tải MỘT LẦN mô hình học sẵn công khai cho khối 2 (cần mạng). Sau đó mọi thứ chạy offline.

    .venv/bin/python scripts/download_models.py            # vinai/phobert-base (giấy phép MIT, 135 triệu tham số)
    .venv/bin/python scripts/download_models.py --all      # thêm uitnlp/visobert, intfloat/multilingual-e5-small (so sánh)
"""
import argparse

from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true")
a = ap.parse_args()
names = ["vinai/phobert-base"] + (["uitnlp/visobert", "intfloat/multilingual-e5-small"] if a.all else [])
for m in names:
    p = snapshot_download(m, allow_patterns=["*.json", "*.model", "*.bin", "*.txt", "*.codes", "*.safetensors"],
                          ignore_patterns=["*flax*", "*tf_model*", "*onnx*"])
    print("đã tải", m, "->", p)
