# Phenikaa Campus Courier — The Codeaholics

Dự đoán bước đi đầu tiên (UP/DOWN/LEFT/RIGHT) của 10 robot giao hàng từ ảnh sơ đồ campus + yêu cầu tiếng Việt
(vòng private test, mã `COURIER2`).

Thành viên: 23010680 – Nguyễn Quốc Thiên · 23010174 – Đỗ Thành Long · 23010069 – Trịnh Gia Huy.

## 1. Cài đặt

Đã kiểm tra trên: Ubuntu, Python 3.14, PyTorch 2.14.1 + CUDA 13.0, GPU RTX 3060 Laptop 6 GB (CPU cũng chạy được, chậm hơn).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Đặt dữ liệu của BTC ở `delivery_v2/` (`train/ validation/ test/ robots.json`).

## 2. Trọng số mô hình

Các mô hình nhỏ đã có sẵn trong repo (`models/`). Hai file lớn hơn 100 MB (giới hạn mỗi file của GitHub) để trên Google Drive:

**Link Google Drive:** https://drive.google.com/drive/folders/1BdmKrAkQdswbQSi7OHmp12A1Ddpw_4nj?usp=sharing

Thư mục Drive `The_Codeaholics_Courier_models/` có cùng cấu trúc với `models/`: chép `nlp_final.pt` vào `models/` và
`phobert-base/pytorch_model.bin` vào `models/phobert-base/` (giữ nguyên tên file).

| tải về rồi đặt vào | dung lượng | sha256 (đầu mã) |
|---|---|---|
| `models/nlp_final.pt` | 538 MB | `eb1f432613631a83…` |
| `models/phobert-base/pytorch_model.bin` | 543 MB | `a0b0f0912c710147…` |

(`phobert-base/` là bản sao cục bộ của `vinai/phobert-base`, giấy phép MIT. Có thể lấy bằng
`.venv/bin/python scripts/download_models.py` rồi chép `pytorch_model.bin` từ `~/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/*/`.)

| file | là gì | tham số |
|---|---|---|
| `kp.pt` `edge.pt` `node.pt` `misc.pt` `config.json` | khối ảnh: dò điểm (ResNet18) + 3 CNN nhỏ trên mảnh ảnh | 5,08 triệu |
| `phobert-base/` | PhoBERT-base gốc (config, tokenizer, trọng số) để chạy offline | (trong dòng dưới) |
| `nlp_final.pt` | PhoBERT tinh chỉnh, 8 đầu ra | 134,52 triệu |
| `nlp_aux_final.pkl` | từ vựng, lưới an toàn n-gram ký tự, n-gram gấp/dễ vỡ | 0,08 triệu |
| `policy_v2.pkl` | robot 1–9: bảng giá có điều kiện + cây tăng cường "chọn bước" | (cây quyết định) |
| **Tổng tham số khi dự đoán** | | **≈ 139,7 triệu** (< 200 triệu) |

## 3. Dự đoán (một lệnh, hoàn toàn offline)

```bash
.venv/bin/python scripts/predict.py --data delivery_v2/test --out predictions.json
sha256sum predictions.json submissions/BEST.json     # hai mã phải giống nhau
```
- Không gọi mạng hay dịch vụ ngoài (tự đặt `HF_HUB_OFFLINE=1`); chỉ đọc `observations.json` + ảnh của thư mục đầu vào.
- 1.200 cảnh mất khoảng 8 phút trên RTX 3060. Kết quả tất định: chạy lại cho file giống hệt.
- `submissions/BEST.json` là file đã nộp (điểm công khai 0,8511), sha256
  `ae0f62710c422663aadb7690a136934ba95e5a2534bd348fa3bd881bfc8c4f54`.

## 4. Huấn luyện lại từ đầu (seed 2026)

```bash
bash scripts/train_all_v2.sh             # ~3 giờ: khối ảnh 3 chặng, khối câu (PhoBERT), khối luật robot
QUICK=1 bash scripts/train_all_v2.sh     # chạy thử nhanh (ít vòng) để kiểm tra không thiếu file -> models_quick/
```
- Cần mạng một lần: PhoBERT (`scripts/download_models.py`) và trọng số ResNet18 ImageNet (torchvision tự tải).
- Riêng khối luật robot (khoảng 15 phút, học lại giống hệt từng byte):
  `COURIER_DIJ=1 COURIER_R9RULE=1 COURIER_EXT=1 COURIER_META_MODES="1:pair,2:deep,6:xdeep,7:big" .venv/bin/python scripts/train_policy.py --final`
- Chỉ dùng dữ liệu được cấp (train + validation). Khối ảnh học trên GPU không tất định tuyệt đối nên học lại cho mô hình
  tương tự, không giống từng byte. Bản nộp dùng đúng trọng số đã lưu ở mục 2.

## 5. Hệ thống

```
ảnh ──khối 1 (src/courier/vision)──► bản đồ: lưới, đoạn đường + trạng thái (theo chú giải của chính ảnh), bậc thang,
                                      một chiều, địa điểm, robot + hướng mũi, mưa/khô, đêm/ngày
câu ──khối 2 (src/courier/nlp)─────► nơi giao (tên / mô tả qua bản đồ), điểm ghé, tham chiếu, gấp, dễ vỡ
bản đồ + phiếu ──khối 3 (src/courier/policy)──► 10 hướng đi
```
Ghép 3 khối: `src/courier/pipeline.py`. Chi tiết phương pháp: file proposal PDF của nhóm.

## 6. Cấu trúc

```
README.md  requirements.txt
src/courier/   data.py graph.py pipeline.py splits.py | vision/ | nlp/ | policy/
scripts/       predict.py | train_all_v2.sh train_vision.py train_nlp_model.py build_nlp_vocab.py build_nlp_aux.py
               train_policy.py download_models.py
models/        trọng số (2 file lớn ở Google Drive, mục 2)
submissions/   BEST.json (file dự đoán đã nộp)
```
