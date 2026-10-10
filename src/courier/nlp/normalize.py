"""Chuẩn hóa câu: chữ thường, bỏ dấu tiếng Việt, gom khoảng trắng."""
import re
import unicodedata


# ký tự "giống hệt" chữ đ/Đ nhưng khác mã (vd Ð của tiếng Iceland), ký tự vô hình, khoảng trắng lạ
_LOOKALIKE = str.maketrans({"Ð": "Đ", "ð": "đ", "Ɖ": "Đ", "ɖ": "đ", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
                            "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\ufeff": "", "\u00ad": ""})


def strip_accents(s):
    s = s.translate(_LOOKALIKE)
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def normalize(text):
    """Trả về câu không dấu, chữ thường, chỉ giữ chữ/số/dấu câu cơ bản."""
    s = strip_accents(unicodedata.normalize("NFKC", text)).lower()
    s = re.sub(r"[^a-z0-9.,!?;:#\[\]\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
