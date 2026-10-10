#!/usr/bin/env bash
# Huấn luyện lại TOÀN BỘ hệ thống V2 từ đầu (seed cố định 2026). Cần GPU ~6 GB, mạng chỉ ở bước 0.
#   bash scripts/train_all_v2.sh            # đầy đủ (~2 giờ)
#   QUICK=1 bash scripts/train_all_v2.sh    # chạy thử nhanh (ít vòng, ít dữ liệu) để chắc không lỗi -> models_quick/
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=models; EP_KP=6; EP_PATCH=12; NLP_ARGS="--n-real 5"
if [ "${QUICK:-0}" = "1" ]; then OUT=models_quick; EP_KP=1; EP_PATCH=1; NLP_ARGS="--n-real 0 --n-synth 200 --epochs 1"; fi
mkdir -p "$OUT"
# 0. mô hình học sẵn công khai (một lần, cần mạng): PhoBERT-base (MIT) -> bản sao cục bộ models/phobert-base
[ -f models/phobert-base/config.json ] || { $PY scripts/download_models.py && $PY - <<'PYEOF'
import glob, shutil, os
snap = [d for d in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/*')) if os.path.exists(d + '/config.json')][0]
os.makedirs('models/phobert-base', exist_ok=True)
for f in ('config.json', 'vocab.txt', 'bpe.codes', 'pytorch_model.bin', 'tokenizer.json'):
    if os.path.exists(f'{snap}/{f}'): shutil.copy(f'{snap}/{f}', 'models/phobert-base/')
PYEOF
}
# 1. khối ảnh (3 chặng, khớp nhật ký logs/train_vision_clean.log, train_vision_vis3.log, train_vision_final.log):
#    (a) từ khởi tạo mặc định (không dùng gì của vòng trước), CHỈ ảnh train V2, tăng cường bản 2
#    (b) học tiếp, tăng cường bản 3 (ảnh xấu nặng hơn)             -> bản dùng cho ước lượng sạch
#    (c) học tiếp trên train + TOÀN BỘ validation (ảnh "khó" giống test) -> bản nộp
TMP_A="$OUT/vis_clean"; TMP_B="$OUT/vis3"
for st in kp patch; do
  if [ $st = kp ]; then EP=$EP_KP; LR=3e-4; else EP=$EP_PATCH; LR=1e-3; fi
  $PY scripts/train_vision.py --stage $st --epochs $EP --aug 2 --lr $LR --workers 6 --out "$TMP_A"
done
for st in kp patch; do
  if [ $st = kp ]; then EP=$(( (EP_KP + 1) * 2 / 3 )); LR=3e-4; else EP=$(( (EP_PATCH + 1) * 2 / 3 )); LR=1e-3; fi
  $PY scripts/train_vision.py --stage $st --epochs $EP --aug 3 --lr $LR --init-from "$TMP_A" --workers 6 --out "$TMP_B"
done
cp "$TMP_A/config.json" "$TMP_B/" 2>/dev/null || true
for st in kp patch; do
  if [ $st = kp ]; then EP=$(( (EP_KP + 1) * 2 / 3 )); LR=3e-4; else EP=$(( (EP_PATCH + 1) * 2 / 3 )); LR=1e-3; fi
  $PY scripts/train_vision.py --stage $st --epochs $EP --aug 3 --lr $LR --init-from "$TMP_B" --use-val-all --workers 6 --out "$OUT"
done
cp "$TMP_B/config.json" "$OUT/" 2>/dev/null || true
# 2. khối câu: bảng món hàng, bộ phận phụ (từ vựng, lưới an toàn, n-gram cờ), PhoBERT chế độ cuối (train + val V2)
$PY scripts/build_nlp_vocab.py
if [ "${QUICK:-0}" != "1" ]; then
  COURIER_NLP_MODE=final $PY scripts/build_nlp_aux.py
  COURIER_NLP_MODE=final $PY scripts/train_nlp_model.py --backbone phobert $NLP_ARGS
  # 3. khối luật: robot 0 là luật tường minh; robot 1-9 mô hình chọn bước (train + toàn bộ validation),
  #    bảng giá có điều kiện học với lộ trình Dijkstra (đi lại ô cũ được) — COURIER_DIJ=số vòng
  #    + kiểu mô hình riêng từng robot (R1 xếp hạng theo cặp, R2 cây 31 lá, R6 đặc trưng v2 + cây 31 lá, R7 200 vòng)
  COURIER_DIJ=${COURIER_DIJ:-1} COURIER_R9RULE=${COURIER_R9RULE:-1} COURIER_EXT=${COURIER_EXT:-1} \
    COURIER_META_MODES=${COURIER_META_MODES:-"1:pair,2:deep,6:xdeep,7:big"} $PY scripts/train_policy.py --final
else
  COURIER_NLP_MODE=final $PY scripts/train_nlp_model.py --backbone phobert --dev $NLP_ARGS
  echo "QUICK: bỏ qua lưu đè models/ cho khối câu và khối luật"
fi
echo "XONG. Dự đoán: .venv/bin/python scripts/predict.py --data delivery_v2/test"
