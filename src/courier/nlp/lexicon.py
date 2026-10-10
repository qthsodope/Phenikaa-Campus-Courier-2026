"""Từ điển cho khối đọc tin nhắn.

Viết có dấu cho dễ đọc; khi nạp sẽ chuẩn hóa về không dấu, chữ thường (normalize.py).
Gồm: tên gọi địa điểm (lấy từ train + họ đồng nghĩa tự bổ sung), cách gọi qua người nhận,
và các cụm từ chỉ vai trò / mức gấp / dễ vỡ / tham chiếu không gian.
"""
from .normalize import normalize

# ----------------------------------------------------------------- địa điểm
PLACES = {
    "library": [
        "phòng sách", "tủ sách", "khu sách", "góc đọc sách", "phòng tra cứu", "kho giáo trình", "trung tâm thư viện", "phòng tư liệu", "phòng đọc tài liệu", "cô thủ thư", "người quản lý thư viện", "nhân viên phòng đọc",
        "phòng tài liệu", "nơi mượn sách tham khảo", "nơi mượn sách", "nơi đọc sách", "khu tài liệu", "kho tài liệu", "nơi tra cứu tài liệu", "phòng mượn sách",
        "thư viện", "thư viện trường", "phòng đọc", "phòng đọc sách", "khu đọc sách", "nơi mượn giáo trình",
        "chỗ trả sách", "nơi trả sách", "chỗ mượn sách", "nơi mượn sách", "quầy mượn sách", "kho sách",
        "trung tâm học liệu", "trung tâm thông tin thư viện", "nhà sách của trường", "khu tự học",
        # người nhận
        "thủ thư", "cán bộ thư viện", "nhân viên thư viện",
    ],
    "dorm": [
        "nhà ký túc", "kí túc xá", "kí túc", "khu kí túc xá", "ký túc xá sinh viên", "khu nhà ở sinh viên", "tòa nội trú", "khu nội trú sinh viên", "chỗ ngủ của sinh viên", "quản lý khu nội trú", "trưởng ban ký túc xá", "sinh viên ở ký túc",
        "dãy phòng nội trú", "khu ở của sinh viên", "dãy nội trú", "khu nhà nội trú", "nơi ở của sinh viên", "nơi sinh viên ở", "dãy ký túc",
        "ký túc xá", "ký túc", "ktx", "khu ktx", "khu ký túc xá", "khu ký túc", "khu nội trú", "nhà nội trú",
        "tòa nhà ở của sinh viên", "phòng ở sinh viên", "phòng ở của sinh viên", "nhà ở sinh viên",
        "nhà ở của sinh viên", "khu ở sinh viên", "khu nhà ở sinh viên", "khu lưu trú", "chỗ ở sinh viên",
        "chỗ ở của sinh viên", "khu nhà ở",
        "ban quản lý ký túc xá", "sinh viên nội trú", "quản lý ktx", "quản lý ký túc xá",
    ],
    "sports": [
        "sân thể dục", "nhà thể dục", "phòng thể chất", "sân điền kinh", "bể bơi", "hồ bơi", "nhà tập luyện", "câu lạc bộ thể thao", "khu tập thể dục", "sân chơi thể thao", "thầy thể chất", "đội bóng rổ", "đội bóng chuyền", "vận động viên", "câu lạc bộ bóng đá",
        "sân luyện tập", "khu vận động", "nơi luyện tập", "nơi tập luyện", "khu luyện tập", "sân vận động", "nơi chơi thể thao", "nhà luyện tập",
        "nhà thể thao", "sân thể thao", "nhà thể chất", "khu thể thao", "sân tập", "sân bóng", "nhà thi đấu",
        "sân vận động", "nhà tập", "nhà đa năng", "khu thể chất", "phòng gym", "phòng tập", "sân cỏ",
        "nhà tập thể thao", "khu tập luyện",
        "huấn luyện viên", "đội bóng", "giáo viên thể chất", "thầy thể dục", "đội tuyển",
    ],
    "clinic": [
        "phòng thuốc", "trạm cấp cứu", "phòng điều trị", "bệnh viện trường", "trung tâm y tế", "phòng chăm sóc sức khỏe", "cô y tá", "bác sĩ trực", "nhân viên y tế trường", "y tá trường",
        "nơi khám sức khỏe", "phòng bác sĩ", "nơi khám bệnh", "phòng khám bệnh", "nơi sơ cứu", "khu khám bệnh",
        "trạm y tế", "phòng y tế", "khu y tế", "phòng khám của trường", "phòng khám", "phòng sơ cứu",
        "trạm xá", "trạm sơ cứu", "bệnh xá", "y tế trường", "phòng khám đa khoa", "nhà y tế",
        "nhân viên y tế", "bác sĩ trường", "bác sĩ", "y tá", "điều dưỡng",
    ],
    "canteen": [
        "nhà bếp", "quán cơm", "căn teen", "khu ăn", "quầy cơm", "nhà ăn tập thể", "bếp tập thể", "phòng ăn sinh viên", "khu ăn trưa", "đầu bếp", "cô bán cơm", "nhân viên nhà ăn", "nhân viên bếp",
        "nơi phục vụ bữa trưa", "quầy ăn", "phòng ăn tập thể", "nhân viên quầy ăn", "nơi ăn uống", "nơi ăn trưa", "khu nhà ăn",
        "căn tin", "căng tin", "canteen", "cantin", "căn-tin", "bếp ăn", "khu ăn uống", "nhà ăn", "phòng ăn",
        "khu ẩm thực", "nhà ăn sinh viên", "bếp trường", "quán ăn của trường", "khu bếp",
        "bếp trưởng", "cô cấp dưỡng", "nhân viên căn tin", "cô nấu ăn",
    ],
    "parking": [
        "nhà gửi xe", "bãi giữ xe", "điểm gửi xe", "khu giữ xe", "chỗ giữ xe", "tầng hầm", "nhà để xe máy", "hầm giữ xe", "bãi để xe", "người trông xe", "bác trông xe", "chú trông xe", "bảo vệ nhà xe",
        "nơi giữ xe", "khu đậu xe", "nơi để xe", "nơi gửi xe", "nơi đỗ xe", "chỗ đậu xe",
        "bãi xe", "bãi đỗ xe", "chỗ để xe", "khu gửi xe", "hầm xe", "nhà xe", "bãi gửi xe", "nhà để xe",
        "hầm gửi xe", "bãi đậu xe", "chỗ gửi xe", "hầm để xe", "khu để xe", "chỗ đỗ xe", "khu đỗ xe",
        "bác giữ xe", "nhân viên trông xe", "chú giữ xe", "bảo vệ bãi xe",
    ],
    "lecture": [
        "tòa học", "khu học", "phòng giảng", "phòng học chung", "hội trường lớn", "giảng đường chính", "thầy đang giảng",
        "nơi lên lớp", "dãy phòng học", "nơi học", "khu phòng học", "phòng lên lớp",
        "giảng đường", "khu giảng đường", "tòa giảng đường", "phòng học lớn", "hội trường học", "lớp học",
        "phòng học", "hội trường", "nhà học", "khu lớp học", "dãy phòng học", "giảng đường lớn",
        "tòa nhà học", "khu học tập",
        "giảng viên đang dạy", "lớp trưởng", "giảng viên",
    ],
    "lab": [
        "phòng lab hóa", "lab sinh", "lab lý", "phòng thực tập", "xưởng cơ khí", "trung tâm thực hành", "phòng thí nghiệm sinh", "nghiên cứu sinh", "trợ giảng phòng lab", "thầy hướng dẫn thí nghiệm",
        "nơi làm thí nghiệm", "phòng thực nghiệm", "khu thực nghiệm", "nơi thực hành",
        "phòng thí nghiệm", "phòng lab", "khu thí nghiệm", "lab hóa", "phòng thực hành", "xưởng thực hành",
        "lab", "phòng nghiên cứu", "khu thực hành", "phòng thí nghiệm hóa", "trung tâm thí nghiệm",
        "xưởng", "khu nghiên cứu",
        "nhóm nghiên cứu", "kỹ thuật viên phòng lab", "kỹ thuật viên", "nghiên cứu viên",
    ],
    "office": [
        "phòng tổ chức", "phòng tài vụ", "phòng kế toán", "văn phòng hành chính", "phòng quản lý", "phòng văn thư", "tòa hành chính", "tòa nhà hành chính", "ban hành chính", "cán bộ văn phòng", "thư ký", "nhân viên hành chính", "kế toán", "văn thư",
        "nơi nộp hồ sơ", "phòng giáo vụ", "nơi làm thủ tục", "nơi làm giấy tờ", "khu giáo vụ",
        "phòng hành chính", "văn phòng khoa", "tòa hiệu bộ", "phòng một cửa", "phòng đào tạo",
        "khu hành chính", "văn phòng", "phòng giáo vụ", "ban giám hiệu", "văn phòng trường", "nhà hiệu bộ",
        "phòng công tác sinh viên", "khu văn phòng",
        "thư ký khoa", "cán bộ phòng đào tạo", "cán bộ hành chính", "giáo vụ",
    ],
    "gate": [
        "cổng lớn", "cổng trường chính", "điểm đón khách", "trạm bảo vệ", "bốt bảo vệ", "phòng thường trực", "lối ra vào", "chú bảo vệ", "bác bảo vệ", "lễ tân", "người gác cổng",
        "nơi khách đến trường", "chốt cổng", "nơi đón khách", "cổng ra vào trường", "chỗ đón khách",
        "cổng trường", "cổng chính", "cổng vào", "cổng a", "chốt bảo vệ cổng", "lối vào trường", "lối vào",
        "cổng", "cổng phụ", "phòng bảo vệ", "chốt bảo vệ", "cổng ra vào",
        "bảo vệ cổng", "tổ đón khách", "bảo vệ",
    ],
}


# ----------------------------------------------------------------- món hàng có chứa tên địa điểm
# Che đi trước khi tìm địa điểm (vd "phong bì công văn" không phải "cổng").
ITEMS = [
    "thẻ thư viện", "sách thư viện", "phong bì công văn", "công văn", "túi sơ cứu", "bộ sơ cứu",
    "bộ dụng cụ thí nghiệm", "dụng cụ thí nghiệm", "mẫu thí nghiệm", "thẻ giữ xe", "vé gửi xe", "chìa khóa xe",
    "bơm xe", "mũ bảo hiểm", "khay cơm", "đồ ăn căn tin", "thẻ ra vào cổng", "thẻ ra vào", "hồ sơ phòng đào tạo",
    "giấy y tế", "hộp y tế", "dụng cụ y tế", "thuốc y tế", "đồ thể thao", "dụng cụ thể thao", "áo thể thao",
    "bảng tên khách", "biển tên khách", "sổ ghi khách", "đồng phục bảo vệ",
]

# ----------------------------------------------------------------- vai trò
# Câu/ mệnh đề chứa các cụm này -> địa điểm trong đó là GÂY NHIỄU (không phải đích, không phải điểm ghé)
DISTRACT = [
    "không cần ghé", "không cần qua", "không cần tới", "không cần đến", "không cần ghé qua", "không phải ghé",
    "không ghé", "đừng ghé", "khỏi ghé", "khỏi cần ghé", "khỏi qua", "bỏ qua", "đừng nhầm", "không nhầm",
    "chớ nhầm", "kẻo nhầm", "nhầm với", "chứ không phải", "không phải là", "không phải", "không giao",
    "đừng giao", "đừng mang", "đừng đưa", "đừng tới", "đừng đến", "đã rời", "rời khỏi", "đã chuyển khỏi",
    "đã đi khỏi", "không còn ở", "không ở", "hôm qua đã giao", "đã giao ở", "đã giao tới", "đã giao đến",
    "hôm trước đã giao", "lần trước đã giao", "tuần trước đã giao", "trước đây đã giao", "đã nhận rồi",
    "không cần", "không phải đi", "tránh nhầm", "không phải điểm nhận", "không phải nơi nhận",
    "không phải ở đó", "không phải chỗ đó", "không phải nơi giao", "không phải điểm giao",
]
# Mệnh đề chứa các cụm này -> địa điểm là ĐIỂM GHÉ (trừ khi là "trước khi ..." = mệnh đề nơi giao)
VIA = [
    "ghé", "tạt qua", "tạt vào", "ghé qua", "ghé ngang", "ghé vào", "trước tiên", "đầu tiên", "trước đã",
    "trước hết", "thứ nhất", "rẽ qua", "ngang qua", "dừng ở", "dừng lại ở", "qua lấy", "qua nhận",
]
# "lấy/nhận ... trước" cũng là điểm ghé (nhưng "cần nhận" là nơi giao)
VIA_TAKE = ["lấy", "nhận", "lấy hàng", "nhận hàng"]
VIA_BEFORE = ["trước"]
GOAL_CLAUSE = ["trước khi", "sau đó", "rồi mới", "cuối cùng", "sau khi"]

# ----------------------------------------------------------------- gấp / dễ vỡ
URGENT_NEG = [
    "không gấp", "không vội", "chẳng gấp", "chả gấp", "đừng vội", "không cần gấp", "không cần vội", "cứ từ từ",
    "từ từ", "thong thả", "chiều nay giao cũng được", "giao cũng được", "lúc nào cũng được", "khi nào cũng được",
    "mai giao cũng được", "không cần nhanh", "không khẩn", "không khẩn cấp", "chưa cần gấp", "chưa gấp",
    "không có gì gấp", "không cần giao ngay", "không cần ngay", "cũng kịp",
    "không cần vội", "chẳng vội", "chả vội", "để sau cũng được",
]
URGENT_POS = [
    "hỏa tốc", "khẩn cấp", "khẩn", "gấp", "cần ngay", "ngay trong", "càng nhanh càng tốt", "nhanh lên",
    "ngay lập tức", "lập tức", "giao ngay", "đi ngay", "ưu tiên", "sớm nhất", "nhanh nhất", "gấp rút",
    "trong 5 phút", "chậm trễ", "không được trễ", "đừng để trễ",
    "ưu tiên số một", "đi liền", "khẩn trương", "tức tốc", "càng sớm càng tốt", "sớm nhất có thể",
    "giao sớm", "nhanh chóng", "mau lên", "đi mau", "giao liền", "cấp tốc",
]
FRAGILE_NEG = [
    "không dễ vỡ", "không vỡ được", "không vỡ", "chắc chắn", "không sợ vỡ", "không lo vỡ", "không dễ hỏng",
    "không bể", "không dễ bể", "không phải đồ dễ vỡ", "đồ bền", "rất bền", "không cần nhẹ tay",
    "không cần cẩn thận", "chịu va đập", "không phải hàng dễ vỡ", "không hề dễ vỡ",
    "rơi cũng chẳng sao", "rơi cũng không sao", "rơi cũng được", "hàng bền", "đồ bền", "không sợ va đập",
    "va đập cũng không sao", "không lo bể",
]
FRAGILE_POS = [
    "dễ vỡ", "dễ hỏng", "dễ bể", "thủy tinh", "đồ gốm", "gốm", "đồ sứ", "nhẹ tay", "tránh va chạm", "va đập",
    "cẩn thận", "mong manh", "dễ nứt", "nâng niu", "va chạm", "kỵ va đập", "dễ móp", "đồ điện tử nhạy",
]

# ----------------------------------------------------------------- tham chiếu không gian
DIRECTIONS = {
    "north": ["phía bắc bản đồ", "phía bắc", "hướng bắc", "mạn bắc", "miền bắc", "phía trên", "mạn trên", "bên trên", "phía trên cùng",
              "trên cùng", "ở trên", "đầu trên", "nửa trên", "phần trên", "phía trên bản đồ", "bên trên bản đồ"],
    "south": ["phía dưới bản đồ", "bên dưới bản đồ", "phía nam bản đồ", "phía nam", "hướng nam", "mạn nam", "miền nam", "phía dưới", "mạn dưới", "bên dưới", "dưới cùng",
              "ở dưới", "đầu dưới", "nửa dưới", "phần dưới"],
    "west": ["bên trái bản đồ", "phía trái bản đồ", "mạn trái bản đồ", "phía tây bản đồ", "phía tây", "hướng tây", "mạn tây", "bên trái", "phía trái", "mạn trái", "trái bản đồ", "nửa trái",
             "phần trái", "bên tay trái"],
    "east": ["bên phải bản đồ", "phía phải bản đồ", "mạn phải bản đồ", "phía đông bản đồ", "phía đông", "hướng đông", "mạn đông", "bên phải", "phía phải", "mạn phải", "phải bản đồ", "nửa phải",
             "phần phải", "bên tay phải"],
}
NEAR = ["gần", "sát", "cạnh", "kế bên", "bên cạnh", "gần với", "gần sát"]
FAR = ["xa", "cách xa", "xa với"]

# ----------------------------------------------------------------- đoán địa điểm theo từ khóa
# Dùng khi gặp tên gọi chưa có trong từ điển: "<từ đầu> ... <từ khóa>", vd "nơi phục vụ bữa trưa".
PLACE_HEADS = ["nơi", "khu", "chỗ", "dãy", "phòng", "tòa", "nhà", "sân", "quầy", "trạm", "bãi", "hầm", "cổng",
               "chốt", "lối", "văn phòng", "xưởng", "góc", "điểm", "trung tâm", "khu vực", "ban", "tổ", "bộ phận"]
PLACE_KEYWORDS = {
    "library": ["sách", "đọc", "tài liệu", "giáo trình", "thư viện", "học liệu", "tra cứu", "mượn"],
    "dorm": ["nội trú", "ký túc", "ktx", "ở của sinh viên", "sinh viên ở", "lưu trú", "ngủ", "nghỉ của sinh viên"],
    "sports": ["thể thao", "thể chất", "vận động", "luyện tập", "tập luyện", "bóng", "thi đấu", "gym", "thể dục",
               "đá bóng", "bơi"],
    "clinic": ["y tế", "khám", "sức khỏe", "sơ cứu", "bệnh", "thuốc", "bác sĩ", "y tá", "chữa", "cấp cứu"],
    "canteen": ["ăn", "bữa", "cơm", "ẩm thực", "nấu", "bếp", "giải khát", "căn tin", "căng tin", "canteen"],
    "parking": ["xe", "đậu xe", "đỗ xe", "gửi xe", "giữ xe"],
    "lecture": ["học", "lớp", "giảng", "lên lớp", "hội trường", "dạy"],
    "lab": ["thí nghiệm", "thực nghiệm", "thực hành", "lab", "nghiên cứu", "hóa chất"],
    "office": ["hành chính", "giáo vụ", "hồ sơ", "hiệu bộ", "đào tạo", "giấy tờ", "thủ tục", "một cửa", "công văn"],
    "gate": ["cổng", "khách đến", "ra vào", "bảo vệ", "lối vào", "đón khách", "tiếp khách"],
}
# từ dừng: gặp là kết thúc cụm tên địa điểm
PLACE_STOP = ["ở", "nằm", "gần", "xa", "sát", "cạnh", "phía", "mạn", "bên", "giúp", "lấy", "nhận", "trước",
              "xong", "để", "và", "rồi", "đang", "cần", "không", "hơn", "cách", "là", "thì", "nhé", "nha"]


# ----------------------------------------------------------------- chế độ (bước 5B, xem mode.py)
# blind: chỉ giữ cụm từ có xuất hiện trong câu TRAIN (gỡ các họ rút ra từ val-A) + danh sách động não.
# final: toàn bộ từ điển + danh sách động não.
def _apply_mode():
    from . import mode, brainstorm as BS
    g = globals()
    keep = lambda x: True  # noqa: E731
    if mode.MODE == "blind" and mode.BLIND_FILTER_V1:
        import re
        corpus = " " + " ".join(" ".join(re.findall(r"[a-z0-9]+", normalize(t))) for t in mode.train_texts()) + " "
        keep = lambda x: (" " + " ".join(re.findall(r"[a-z0-9]+", normalize(x))) + " ") in corpus  # noqa: E731
    bs = mode.BRAINSTORM

    def merge(own, extra):
        # Chống "nhiễm": danh sách động não được soạn SAU khi người soạn đã thấy từ điển cũ (có họ rút từ val-A).
        # Ở chế độ mù, cụm động não nào trùng một cụm của từ điển cũ KHÔNG có trong train cũng bị gỡ (đo bi quan).
        suspect = {normalize(x) for x in own if not keep(x)}
        extra = [x for x in extra if normalize(x) not in suspect] if (mode.MODE == "blind" and mode.BLIND_FILTER_V1) else list(extra)
        out = [x for x in own if keep(x)] + (extra if bs else [])
        return list(dict.fromkeys(out))

    for name in ("ITEMS", "DISTRACT", "VIA", "VIA_TAKE", "GOAL_CLAUSE", "URGENT_NEG", "URGENT_POS", "FRAGILE_NEG",
                 "FRAGILE_POS", "NEAR", "FAR", "PLACE_HEADS", "PLACE_STOP"):
        extra = list(getattr(BS, name, []))
        if mode.FLAG_V2:
            extra += getattr(BS, name + "_2", [])
        g[name] = merge(g[name], extra)
    for name in ("PLACES", "DIRECTIONS", "PLACE_KEYWORDS"):
        g[name] = {k: merge(v, getattr(BS, name).get(k, [])) for k, v in g[name].items()}


_apply_mode()

# Từ khóa gọn để in ra; khóa đúng như scenes.json
PLACE_TYPES = list(PLACES)


def _norm_list(xs):
    return sorted({normalize(x) for x in xs}, key=lambda s: (-len(s), s))


ALIASES = []          # [(alias không dấu, loại)] — dài trước
for _t, _xs in PLACES.items():
    for _a in _norm_list(_xs):
        ALIASES.append((_a, _t))
ALIASES.sort(key=lambda p: (-len(p[0]), p[0], p[1]))
