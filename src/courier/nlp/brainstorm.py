"""DANH SÁCH ĐỘNG NÃO cách nói tiếng Việt (bước 5B).

NGUỒN: tự động não (Claude, 07/10/2026) theo hiểu biết chung về tiếng Việt đời thường ở trường đại học.
KHÔNG lấy từ câu validation hay test (không mở, không đọc câu nào của hai tập này khi soạn).
Ghi chú trung thực: người soạn đã từng thấy từ điển cũ (lexicon.py), trong đó có một số "họ" cách nói rút ra từ
val-A ở bước 3, nên danh sách này có thể trùng một phần với các họ đó.

Dùng cho: (1) mở rộng từ điển ở mọi chế độ; (2) sinh câu luyện tập cho mô hình (thay tên, bỏ dấu, gõ sai...).
Viết có dấu; khi nạp sẽ được chuẩn hóa về không dấu (normalize.py), nên tự động chịu được câu không dấu.
"""

# ----------------------------------------------------------------- tên gọi 10 loại địa điểm
# Theo: tên chính thức, tên dân dã, viết tắt, từ mượn, chức năng ("nơi + hoạt động"), đồ vật đặc trưng,
# người làm việc ở đó (người nhận).
PLACES = {
    "library": [
        "thư viện", "thư viện trường", "thư viện trung tâm", "thư viện chính", "tv trường", "phòng đọc",
        "phòng đọc chung", "phòng đọc sách", "khu đọc sách", "góc đọc", "góc sách", "kho sách", "tủ sách",
        "nhà sách trường", "thư quán", "thư khố", "trung tâm học liệu", "trung tâm thông tin thư viện",
        "trung tâm tư liệu", "phòng tư liệu", "phòng tài liệu", "kho tài liệu", "kho giáo trình", "kho lưu trữ sách",
        "phòng lưu trữ tài liệu", "khu tự học", "phòng tự học", "phòng mượn trả sách", "quầy mượn trả sách",
        "quầy mượn sách", "quầy trả sách", "nơi mượn sách", "nơi trả sách", "nơi mượn giáo trình", "nơi đọc sách",
        "nơi tra cứu tài liệu", "nơi tra cứu", "nơi cho mượn sách", "chỗ mượn sách", "chỗ đọc sách", "chỗ trả sách",
        "library", "phòng library", "khu library", "ngôi nhà tri thức",
        "thủ thư", "cô thủ thư", "chú thủ thư", "anh thủ thư", "chị thủ thư", "cán bộ thư viện", "nhân viên thư viện",
        "người quản lý thư viện", "quản lý thư viện", "nhân viên phòng đọc", "cô phụ trách thư viện",
    ],
    "dorm": [
        "ký túc xá", "kí túc xá", "ký túc", "kí túc", "ktx", "khu ktx", "khu ký túc", "khu ký túc xá", "tòa ktx",
        "nhà ktx", "dãy ktx", "dãy ký túc", "khu nội trú", "nhà nội trú", "dãy nội trú", "tòa nội trú", "khu lưu trú",
        "nhà lưu trú", "khu nhà ở sinh viên", "nhà ở sinh viên", "khu ở sinh viên", "chỗ ở sinh viên",
        "chỗ ở của sinh viên", "nơi ở sinh viên", "nơi ở của sinh viên", "nơi sinh viên ở", "nơi sinh viên ngủ",
        "chỗ ngủ sinh viên", "khu phòng ở", "dãy phòng ở", "phòng ở sinh viên", "phòng trọ trong trường",
        "khu trọ sinh viên", "khu ở nội trú", "làng sinh viên", "khu tập thể sinh viên", "dorm", "khu dorm",
        "nơi nghỉ ngơi của sinh viên", "nhà ở tập thể",
        "quản lý ktx", "quản lý ký túc xá", "ban quản lý ký túc xá", "ban quản lý ktx", "cô quản lý ký túc",
        "chú quản lý ký túc", "sinh viên nội trú", "sinh viên ở ký túc", "sinh viên ktx", "trưởng phòng ký túc",
        "trưởng tòa ktx", "bác quản lý khu nội trú",
    ],
    "sports": [
        "nhà thể thao", "nhà thể chất", "nhà thi đấu", "nhà đa năng", "nhà tập", "nhà tập luyện", "nhà luyện tập",
        "sân thể thao", "sân thể dục", "sân vận động", "svđ", "sân bóng", "sân bóng đá", "sân bóng rổ",
        "sân bóng chuyền", "sân cầu lông", "sân tennis", "sân quần vợt", "sân điền kinh", "sân cỏ", "sân tập",
        "sân luyện tập", "đường chạy", "bể bơi", "hồ bơi", "phòng gym", "phòng tập gym", "gym", "khu gym",
        "phòng tập", "phòng thể chất", "phòng thể dục", "khu thể thao", "khu thể chất", "khu vận động", "khu tập luyện",
        "khu luyện tập", "trung tâm thể thao", "trung tâm thể dục thể thao", "câu lạc bộ thể thao", "nơi tập thể dục",
        "nơi chơi thể thao", "nơi tập thể thao", "nơi luyện tập", "nơi tập luyện", "nơi đá bóng", "nơi chơi bóng",
        "chỗ tập thể dục", "chỗ chơi thể thao", "chỗ đá bóng", "sân chơi thể thao", "khu sân bãi",
        "huấn luyện viên", "hlv", "thầy thể dục", "cô thể dục", "thầy thể chất", "cô thể chất", "giáo viên thể chất",
        "giáo viên thể dục", "giảng viên thể chất", "đội bóng", "đội bóng đá", "đội bóng rổ", "đội bóng chuyền",
        "đội tuyển", "đội tuyển trường", "vận động viên", "câu lạc bộ bóng đá", "clb bóng đá", "clb thể thao",
    ],
    "clinic": [
        "trạm y tế", "phòng y tế", "y tế trường", "khu y tế", "nhà y tế", "trung tâm y tế", "phòng khám",
        "phòng khám bệnh", "phòng khám đa khoa", "phòng khám của trường", "phòng khám trường", "bệnh xá", "trạm xá",
        "bệnh viện trường", "phòng sơ cứu", "trạm sơ cứu", "trạm cấp cứu", "phòng cấp cứu", "phòng điều trị",
        "phòng thuốc", "tủ thuốc", "quầy thuốc", "phòng chăm sóc sức khỏe", "phòng bác sĩ", "phòng y tá",
        "nơi khám bệnh", "nơi khám sức khỏe", "nơi chữa bệnh", "nơi sơ cứu", "nơi cấp thuốc", "nơi phát thuốc",
        "nơi lấy thuốc", "chỗ khám bệnh", "chỗ lấy thuốc", "khu khám bệnh", "phòng khám sức khỏe",
        "bác sĩ", "bác sĩ trường", "bác sĩ trực", "y tá", "cô y tá", "y tá trường", "điều dưỡng", "cô điều dưỡng",
        "nhân viên y tế", "nhân viên y tế trường", "cán bộ y tế", "y sĩ", "dược sĩ",
    ],
    "canteen": [
        "căn tin", "căng tin", "căn-tin", "cantin", "canteen", "căn teen", "nhà ăn", "nhà ăn sinh viên",
        "nhà ăn tập thể", "phòng ăn", "phòng ăn tập thể", "bếp ăn", "bếp ăn tập thể", "bếp tập thể", "bếp trường",
        "khu bếp", "nhà bếp", "khu ăn", "khu ăn uống", "khu ăn trưa", "khu ẩm thực", "khu nhà ăn", "quán ăn",
        "quán ăn của trường", "quán cơm", "quầy cơm", "quầy ăn", "quầy đồ ăn", "food court", "foodcourt",
        "nơi ăn uống", "nơi ăn trưa", "nơi ăn cơm", "nơi bán đồ ăn", "nơi phục vụ bữa trưa", "nơi phục vụ bữa ăn",
        "chỗ ăn trưa", "chỗ ăn cơm", "chỗ ăn uống", "chỗ bán cơm", "phòng ăn sinh viên", "quầy giải khát",
        "nhà ăn chung", "bếp ăn chung", "bếp chung",
        "đầu bếp", "bếp trưởng", "cô bếp", "cô cấp dưỡng", "cô nấu ăn", "cô bán cơm", "nhân viên căn tin",
        "nhân viên nhà ăn", "nhân viên bếp", "nhân viên quầy ăn", "chủ căn tin", "cô căn tin", "người bán cơm",
    ],
    "parking": [
        "bãi xe", "bãi đỗ xe", "bãi đậu xe", "bãi gửi xe", "bãi giữ xe", "bãi để xe", "nhà xe", "nhà để xe",
        "nhà gửi xe", "nhà để xe máy", "hầm xe", "hầm gửi xe", "hầm để xe", "hầm giữ xe", "hầm đỗ xe", "tầng hầm",
        "tầng hầm để xe", "khu gửi xe", "khu để xe", "khu đỗ xe", "khu đậu xe", "khu giữ xe", "điểm gửi xe",
        "điểm đỗ xe", "chỗ để xe", "chỗ gửi xe", "chỗ đỗ xe", "chỗ đậu xe", "chỗ giữ xe", "nơi gửi xe", "nơi để xe",
        "nơi đỗ xe", "nơi đậu xe", "nơi giữ xe", "bến xe trường", "lán xe", "lán để xe", "nhà xe sinh viên",
        "parking", "bãi parking", "khu parking",
        "bác giữ xe", "chú giữ xe", "bác trông xe", "chú trông xe", "người trông xe", "người giữ xe",
        "nhân viên trông xe", "nhân viên giữ xe", "bảo vệ nhà xe", "bảo vệ bãi xe", "cô giữ xe",
    ],
    "lecture": [
        "giảng đường", "khu giảng đường", "tòa giảng đường", "giảng đường lớn", "giảng đường chính", "nhà giảng đường",
        "hội trường", "hội trường lớn", "hội trường học", "phòng học", "phòng học lớn", "phòng học chung",
        "dãy phòng học", "khu phòng học", "lớp học", "khu lớp học", "nhà học", "tòa học", "tòa nhà học",
        "khu học", "khu học tập", "phòng giảng", "phòng lên lớp", "nơi lên lớp", "nơi học", "nơi học tập",
        "nơi giảng dạy", "chỗ học", "chỗ lên lớp", "giảng đường a", "phòng chuyên đề",
        "phòng seminar", "khu giảng dạy",
        "giảng viên", "giảng viên đang dạy", "thầy đang giảng", "cô đang giảng",
        "lớp trưởng", "sinh viên trong lớp", "lớp đang học",
    ],
    "lab": [
        "phòng thí nghiệm", "phòng tn", "phòng thí nghiệm hóa", "phòng thí nghiệm sinh", "phòng thí nghiệm lý",
        "phòng lab", "lab", "lab hóa", "lab sinh", "lab lý", "lab máy tính", "phòng máy", "khu thí nghiệm",
        "trung tâm thí nghiệm", "phòng thực hành", "khu thực hành", "xưởng thực hành", "xưởng", "xưởng cơ khí",
        "phòng thực tập", "trung tâm thực hành", "phòng thực nghiệm", "khu thực nghiệm", "phòng nghiên cứu",
        "khu nghiên cứu", "viện nghiên cứu", "trung tâm nghiên cứu", "nơi làm thí nghiệm", "nơi thực hành",
        "nơi nghiên cứu", "chỗ làm thí nghiệm", "chỗ thực hành", "phòng hóa nghiệm", "phòng thí nghiệm trọng điểm",
        "kỹ thuật viên", "kỹ thuật viên phòng lab", "kỹ thuật viên thí nghiệm", "nghiên cứu viên", "nghiên cứu sinh",
        "nhóm nghiên cứu", "trợ giảng phòng lab", "thầy hướng dẫn thí nghiệm", "người phụ trách phòng lab",
    ],
    "office": [
        "văn phòng", "văn phòng trường", "văn phòng khoa", "văn phòng hành chính", "phòng hành chính",
        "khu hành chính", "tòa hành chính", "tòa nhà hành chính", "nhà hành chính", "ban hành chính", "tòa hiệu bộ",
        "nhà hiệu bộ", "khu hiệu bộ", "ban giám hiệu", "phòng đào tạo", "phòng giáo vụ", "khu giáo vụ",
        "phòng công tác sinh viên", "phòng ctsv", "phòng một cửa", "bộ phận một cửa", "phòng tổ chức",
        "phòng tài vụ", "phòng kế toán", "phòng tài chính", "phòng văn thư", "phòng quản lý", "khu văn phòng",
        "nơi nộp hồ sơ", "nơi làm thủ tục", "nơi làm giấy tờ", "nơi đóng dấu", "nơi xin giấy xác nhận",
        "chỗ nộp hồ sơ", "chỗ làm thủ tục", "phòng tiếp nhận hồ sơ",
        "thư ký", "thư ký khoa", "văn thư", "kế toán", "giáo vụ", "cán bộ giáo vụ", "cán bộ văn phòng",
        "cán bộ hành chính", "nhân viên hành chính", "nhân viên văn phòng", "cán bộ phòng đào tạo", "thầy hiệu trưởng",
        "hiệu trưởng", "phó hiệu trưởng",
    ],
    "gate": [
        "cổng", "cổng trường", "cổng chính", "cổng lớn", "cổng phụ", "cổng vào", "cổng ra vào", "cổng ra vào trường",
        "cổng trường chính", "cổng sau", "cổng trước", "cổng a", "cổng b", "lối vào", "lối vào trường", "lối ra vào",
        "lối vào chính", "cửa ngõ", "cửa ngõ trường", "phòng bảo vệ", "phòng thường trực", "chốt bảo vệ",
        "chốt bảo vệ cổng", "chốt cổng", "trạm bảo vệ", "bốt bảo vệ", "chốt gác", "trạm gác", "nơi đón khách",
        "chỗ đón khách", "điểm đón khách", "quầy lễ tân", "sảnh đón khách", "nơi khách đến trường",
        "nơi tiếp khách ở cổng", "barie cổng",
        "bảo vệ", "bảo vệ cổng", "chú bảo vệ", "bác bảo vệ", "anh bảo vệ", "người gác cổng", "lễ tân",
        "tổ đón khách", "tổ bảo vệ", "đội bảo vệ",
    ],
}

# Từ khóa để đoán tên gọi lạ: "<từ đầu> ... <từ khóa>" (vd "nơi cấp thuốc" -> clinic)
PLACE_KEYWORDS = {
    "library": ["sách", "đọc", "tài liệu", "giáo trình", "thư viện", "học liệu", "tra cứu", "mượn", "tư liệu",
                "thủ thư", "thư quán", "library"],
    "dorm": ["nội trú", "ký túc", "kí túc", "ktx", "lưu trú", "ngủ", "sinh viên ở", "ở của sinh viên", "chỗ ở",
             "nơi ở", "dorm", "phòng ở"],
    "sports": ["thể thao", "thể chất", "thể dục", "vận động", "luyện tập", "tập luyện", "bóng", "thi đấu", "gym",
               "bơi", "điền kinh", "cầu lông", "tennis", "chạy bộ"],
    "clinic": ["y tế", "khám", "sức khỏe", "sơ cứu", "bệnh", "thuốc", "bác sĩ", "y tá", "chữa", "cấp cứu",
               "điều trị", "điều dưỡng"],
    "canteen": ["ăn", "bữa", "cơm", "ẩm thực", "nấu", "bếp", "giải khát", "căn tin", "căng tin", "canteen",
                "đồ uống", "food"],
    "parking": ["xe", "đậu xe", "đỗ xe", "gửi xe", "giữ xe", "parking"],
    "lecture": ["học", "lớp", "giảng", "lên lớp", "hội trường", "dạy", "seminar"],
    "lab": ["thí nghiệm", "thực nghiệm", "thực hành", "lab", "nghiên cứu", "hóa chất", "xưởng", "thực tập"],
    "office": ["hành chính", "giáo vụ", "hồ sơ", "hiệu bộ", "đào tạo", "giấy tờ", "thủ tục", "một cửa",
               "công văn", "văn phòng", "tài vụ", "kế toán", "đóng dấu", "văn thư"],
    "gate": ["cổng", "khách đến", "ra vào", "bảo vệ", "lối vào", "đón khách", "tiếp khách", "thường trực", "gác"],
}
PLACE_HEADS = ["nơi", "khu", "chỗ", "dãy", "phòng", "tòa", "nhà", "sân", "quầy", "trạm", "bãi", "hầm", "cổng",
               "chốt", "lối", "văn phòng", "xưởng", "góc", "điểm", "trung tâm", "khu vực", "ban", "tổ", "bộ phận",
               "tầng", "sảnh", "lán", "bốt", "viện", "kho", "tủ"]

# ----------------------------------------------------------------- vai trò
DISTRACT = [
    # không cần ghé
    "không cần ghé", "không cần ghé qua", "không cần qua", "không cần tới", "không cần đến", "không cần vào",
    "không phải ghé", "không phải qua", "không phải tới", "không ghé", "đừng ghé", "đừng qua", "đừng tới",
    "đừng đến", "đừng vào", "khỏi ghé", "khỏi qua", "khỏi cần ghé", "khỏi phải ghé", "bỏ qua", "chẳng cần ghé",
    "chả cần ghé", "không cần phải ghé", "không cần ghé ngang", "không cần dừng", "đừng dừng", "không dừng",
    "khong can ghe", "ko cần ghé", "k cần ghé", "kh cần ghé",
    # đừng nhầm
    "đừng nhầm", "không nhầm", "chớ nhầm", "kẻo nhầm", "nhầm với", "tránh nhầm", "coi chừng nhầm", "cẩn thận nhầm",
    "dễ nhầm", "chứ không phải", "không phải là", "không phải", "chẳng phải", "chả phải", "không phải điểm nhận",
    "không phải nơi nhận", "không phải nơi giao", "không phải điểm giao", "không phải ở đó", "không phải chỗ đó",
    # không giao
    "không giao", "đừng giao", "đừng mang", "đừng đưa", "đừng gửi", "không mang", "không đưa", "không gửi",
    # đã rời / đã giao
    "đã rời", "rời khỏi", "vừa rời", "đã đi khỏi", "đã chuyển khỏi", "đã chuyển đi", "không còn ở", "không ở",
    "chuyển khỏi", "đã dọn khỏi", "không còn làm ở", "hôm qua đã giao", "hôm trước đã giao", "lần trước đã giao",
    "tuần trước đã giao", "trước đây đã giao", "sáng nay đã giao", "đã giao ở", "đã giao tới", "đã giao đến",
    "đã giao cho", "đã nhận rồi", "đã lấy rồi", "không cần",
    # hủy / đổi
    "đã hủy", "hủy đơn", "đổi địa chỉ", "không phải đi", 
]
VIA = [
    "ghé", "ghé qua", "ghé ngang", "ghé vào", "ghé tạt", "tạt qua", "tạt vào", "rẽ qua", "rẽ vào", "ngang qua",
    "đi ngang", "dừng ở", "dừng lại ở", "dừng tại", "ghé lấy", "qua lấy", "qua nhận", "vào lấy", "đến lấy",
    "tới lấy", "trước tiên", "đầu tiên", "trước hết", "trước đã", "thứ nhất", "bước một", "bước 1", "trước nhất",
    "đầu tiên là", "tiện đường", "sẵn đường", "thuận đường",
]
VIA_TAKE = ["lấy", "nhận", "lấy hàng", "nhận hàng", "cầm", "mượn", "lấy thêm", "nhận thêm", "gom"]
GOAL_CLAUSE = ["trước khi", "sau đó", "rồi mới", "cuối cùng", "sau khi", "sau cùng", "rồi hãy",
               "điểm cuối", "đích đến", "điểm giao", "nơi giao", "nơi nhận"]

# ----------------------------------------------------------------- gấp / dễ vỡ
URGENT_NEG = [
    "không gấp", "không vội", "chẳng gấp", "chả gấp", "không cần gấp", "không cần vội", "đừng vội", "cứ từ từ",
    "từ từ", "thong thả", "thong dong", "không khẩn", "không khẩn cấp", "chưa gấp", "chưa cần gấp",
    "không có gì gấp", "không cần nhanh", "không cần giao ngay", "không cần ngay", "không cần giao gấp",
    "chẳng vội", "chả vội", "không vội đâu", "không gấp lắm", "không quá gấp", "không cần khẩn trương",
    "lúc nào cũng được", "khi nào cũng được", "bao giờ cũng được", "giao cũng được", "chiều nay giao cũng được",
    "mai giao cũng được", "để sau cũng được", "cũng kịp", "vẫn kịp", "còn thời gian", "dư thời gian",
    "không cần hỏa tốc", "không cần ưu tiên", "rảnh thì đi", "lúc rảnh", "khi rảnh", "không bắt buộc nhanh",
    "ko gấp", "k gấp", "kg gấp",
]
URGENT_POS = [
    "gấp", "rất gấp", "gấp lắm", "gấp rút", "khẩn", "khẩn cấp", "khẩn trương", "hỏa tốc", "cấp tốc", "tức tốc",
    "cần ngay", "ngay trong", "ngay lập tức", "lập tức", "giao ngay", "đi ngay", "mang ngay", "đưa ngay",
    "giao liền", "đi liền", "ngay và luôn", "càng nhanh càng tốt", "càng sớm càng tốt", "sớm nhất", "sớm nhất có thể",
    "nhanh nhất", "nhanh nhất có thể", "nhanh lên", "nhanh chóng", "mau lên", "đi mau", "giao sớm",
    "ưu tiên", "ưu tiên số một", "ưu tiên hàng đầu", "chậm trễ", "không được trễ", "đừng để trễ", "kẻo trễ",
    "kẻo muộn", "không được chậm", "đừng chậm", "trong 5 phút", "trong 10 phút", "trong vài phút", "sắp trễ",
    "đang cần gấp", "cần gấp", "urgent", "asap", "trễ hạn", "sát giờ", 
]
FRAGILE_NEG = [
    "không dễ vỡ", "không vỡ", "không vỡ được", "không bị vỡ", "không sợ vỡ", "không lo vỡ", "không dễ hỏng",
    "không dễ bể", "không bể", "không lo bể", "không phải đồ dễ vỡ", "không phải hàng dễ vỡ", "không hề dễ vỡ",
    "chắc chắn", "rất chắc", "chắc lắm", "đồ bền", "hàng bền", "rất bền", "bền chắc",
    "cứng cáp", "chịu va đập", "không sợ va đập", "va đập cũng không sao", "rơi cũng chẳng sao", "rơi cũng không sao",
    "rơi cũng được", "rơi không sao", "không cần nhẹ tay", "không cần cẩn thận", "không cần nâng niu",
    "không cần nhẹ nhàng", "không có đồ dễ vỡ", "đồ không vỡ", "không phải đồ thủy tinh", "ném cũng được",
    "ko dễ vỡ", "k dễ vỡ",
    # bổ sung (bước 5B, lượt 2)
    "không sợ hỏng", "không lo hỏng", "khó vỡ", "khó bể", "khó hỏng", "không thể vỡ", "chẳng vỡ được",
    "không cần nâng niu gì", "cứ thoải mái",
    "quăng cũng được", "va chạm cũng không sao", "không có gì dễ vỡ", "không có gì dễ hỏng",
]
FRAGILE_POS = [
    "dễ vỡ", "dễ hỏng", "dễ bể", "dễ nứt", "dễ móp", "dễ gãy", "thủy tinh", "đồ gốm", "gốm", "đồ sứ", "pha lê", "nhẹ tay", "nhẹ nhàng", "cẩn thận", "cẩn thận giúp", "nâng niu", "mong manh", "tránh va chạm",
    "va chạm", "va đập", "kỵ va đập", "tránh rung lắc", "rung lắc", "tránh xóc", "đừng làm rơi", "đừng để rơi",
    "đừng làm vỡ", "đồ điện tử nhạy", "hàng nhạy cảm", "fragile", "hàng dễ vỡ", "đồ dễ vỡ",
    # bổ sung (bước 5B, lượt 2): lời dặn cách cầm/chở
    "xin nhẹ tay", "nhẹ tay giúp", "nhẹ tay nhé", "cầm nhẹ", "đặt nhẹ", "để nhẹ", "đi chậm kẻo vỡ", "kẻo vỡ",
    "kẻo bể", "sợ vỡ", "sợ bể", "lo vỡ", "vỡ mất", "vỡ là hỏng", "bể là hỏng", "đừng quăng", "đừng ném",
    "không được ném", "không được quăng", "đừng đè", "không được đè", "đừng chồng đồ lên", "đừng lật",
    "không được lật", "giữ thẳng đứng", "để thẳng đứng", "đừng để va", "không được va",
    "tránh va", "tránh đụng", "đừng để đổ", "tránh rơi", "không được rơi", "chống sốc", "hàng cần chống sốc",
    "dễ trầy", "dễ xước", "dễ méo", "dễ tổn hại", "rất mỏng", "mỏng manh", "dễ gãy vụn",
    "hàng mỏng", "đồ mỏng", "cẩn thận kẻo vỡ",
    "handle with care", "cần nâng niu", "nâng như nâng trứng", "cẩn trọng",
    # chất liệu / đồ vật dễ vỡ
    "đồ thủy tinh", "ly thủy tinh", "cốc thủy tinh", "chai thủy tinh", "bình thủy tinh", "lọ thủy tinh",
    "đồ pha lê", "ly pha lê", "chén sứ", "bát sứ", "đĩa sứ", "bình sứ", "bình gốm", "lọ gốm", "đĩa gốm",
    "đồ gốm sứ", "gốm sứ", "lọ hoa thủy tinh", "tấm kính", "khung kính", "mặt kính",
    "ống nghiệm", "lam kính", "linh kiện nhạy",
    "tượng thạch cao", "mô hình thạch cao", "đồ thủ công", "đồ sành",
]

# ----------------------------------------------------------------- gấp / dễ vỡ: bổ sung bước 5C (phần A)
# NGUỒN: tự động não (Claude, 07/10/2026), không nhìn validation/test. Gồm cách nói trực tiếp và NGẦM (hạn giờ,
# sự kiện sắp diễn ra, người nhận sắp đi), các dạng phủ định. Dùng cho luật và để sinh câu luyện tập.
URGENT_POS_2 = [
    # từ lõi một chữ (đi kèm luật phủ định có phạm vi): kiểm tra trên train chỉ xuất hiện trong câu gấp
    "nhanh", "khẩn", "tốc", "sớm",
    "gấp gáp", "đang vội", "rất vội", "vội lắm", "cần liền", "cần gấp lắm", "cực gấp", "siêu gấp", "gấp gấp",
    "rất khẩn", "khẩn lắm", "cực khẩn", "tốc hành", "siêu tốc", "chuyển phát nhanh", "đơn hỏa tốc", "đơn gấp",
    "đơn khẩn", "hàng ưu tiên", "đơn ưu tiên", "ngay bây giờ", "bây giờ luôn", "ngay tức khắc", "ngay tắp lự",
    "lẹ lên", "nhanh nhanh", "nhanh giùm", "nhanh giúp", "nhanh hộ", "mau mau", "đi nhanh", "chạy nhanh",
    "không chờ được", "không thể chờ", "không đợi được", "chờ không nổi", "đợi không nổi", "không được để lâu",
    "không được chậm trễ", "không được chậm", "đừng chần chừ", "chậm là hỏng việc", "trễ là hỏng việc",
    "không kịp mất", "kẻo không kịp", "kẻo lỡ", "sợ không kịp", "cho kịp", "cho kịp giờ", "phải kịp",
    # hạn giờ / sự kiện (ngầm)
    "sắp đến giờ", "sắp tới giờ", "sắp hết giờ", "sắp họp", "sắp thi", "sắp vào lớp", "sắp vào ca",
    "trước giờ họp", "trước giờ thi", "trước giờ học", "trước giờ vào lớp", "kịp buổi họp", "kịp giờ họp",
    "kịp giờ thi", "kịp giờ học", "kịp buổi học", "kịp buổi thí nghiệm", "cuộc họp sắp bắt đầu",
    "buổi thi sắp bắt đầu", "người nhận sắp về", "người nhận sắp đi", "người nhận sắp rời", "người nhận đang đợi gấp",
    "sắp đóng cửa", "trước khi đóng cửa", "hạn chót", "deadline", "sát nút", "sát giờ", "trong vòng 5 phút",
    "trong 15 phút", "trong mười phút", "trong nửa tiếng", "trong 30 phút", "ngay sáng nay", "ngay trưa nay",
]
URGENT_NEG_2 = [
    "không gấp gáp", "không cần gấp gáp", "không cần vội vàng", "không phải gấp", "không phải việc gấp",
    "không phải đơn gấp", "chẳng cần vội", "đâu có gấp", "đâu có vội", "có gì đâu mà vội", "không việc gì phải vội",
    "từ từ cũng được", "chậm cũng được", "chậm chút cũng được", "muộn chút cũng được", "trễ chút cũng không sao",
    "trễ cũng không sao", "muộn cũng không sao", "chiều giao cũng được", "tối giao cũng được", "mai cũng được",
    "ngày mai cũng được", "tuần sau cũng được", "cuối ngày cũng được", "trong hôm nay là được", "hôm nay là được",
    "lúc nào tiện", "khi nào tiện", "tiện thì giao", "rảnh thì giao", "đơn thường", "giao thường",
    "không ưu tiên", "không có hạn", "không có deadline", "không cần kịp", "thoải mái thời gian", "cứ thong thả",
    "không cần nhanh lắm", "không vội lắm", "chưa cần ngay", "chưa vội", "để lúc khác cũng được",
]
FRAGILE_POS_2 = [
    # từ lõi một chữ (đi kèm luật phủ định có phạm vi); KHÔNG dùng "bể" (trùng "bể bơi"), "hỏng" (trùng "hông")
    "vỡ", "nứt", "móp",
    "dễ vỡ lắm", "rất dễ vỡ", "cực dễ vỡ", "dễ hư", "dễ hư hỏng", "hàng dễ hư", "dễ bẹp", "dễ dập", "dễ biến dạng",
    "đóng gói kỹ", "đi êm", "đi chậm thôi", "chạy êm", "tránh ổ gà", "tránh đường xóc", "đừng lắc", "không được lắc",
    "không được nghiêng", "đừng nghiêng", "giữ nguyên chiều", "hàng tinh xảo", "thiết bị chính xác",
    "dụng cụ chính xác", "mẫu xét nghiệm", "dụng cụ thủy tinh", "bình thủy tinh", "đồ thủy tinh", "hàng thủy tinh",
    "đồ sứ", "hàng gốm", "hàng sứ", "đồ pha lê", "hàng pha lê", "cẩn thận vỡ", "coi chừng vỡ", "coi chừng bể",
    "cẩn thận bể", "lỡ rơi là vỡ", "rơi là vỡ", "rơi là bể", "va là vỡ", "nhẹ tay thôi", "nhẹ nhàng thôi",
]
FRAGILE_NEG_2 = [
    "không dễ hư", "không dễ hỏng đâu", "rất chắc chắn", "chắc nịch", "hàng cứng", "đồ cứng", "rơi không vỡ",
    "có rơi cũng không sao", "rơi cũng chả sao", "không cần đóng gói kỹ", "va chạm thoải mái", "xóc cũng được",
    "lắc cũng được", "không phải đồ dễ hỏng", "chẳng có gì dễ vỡ", "toàn giấy tờ", "chỉ là giấy tờ",
    "chỉ là quần áo", "chỉ là sách", "hàng mềm", "đồ mềm", "bền lắm", "rất bền bỉ", "không lo hư",
    "không sợ hư", "không sợ bể", "không cần cẩn thận lắm", "cứ để thoải mái",
]
# Đồ vật NGẦM dễ vỡ / chắc chắn: CHỈ dùng trong khung câu "Bên trong là ...", "Trong hộp có ..." để dạy mô hình
# (không dùng cho luật, vì món hàng một mình không quyết định nhãn trong dữ liệu).
MATERIALS_FRAGILE = ["trứng", "bánh kem", "gương", "màn hình", "bình hoa", "kính hiển vi", "lọ nước hoa",
                     "ly thủy tinh", "chén sứ", "đĩa gốm", "tượng thạch cao", "ống nghiệm", "bóng đèn tuýp",
                     "lam kính", "mẫu xét nghiệm", "máy ảnh", "ống kính", "đồng hồ thủy tinh", "bể cá mini"]
MATERIALS_STURDY = ["quần áo", "giấy tờ", "sách vở", "đồ nhựa", "chăn gối", "khăn", "giày dép", "bóng đá",
                    "hồ sơ", "vải vóc", "đồ len"]

# ----------------------------------------------------------------- tham chiếu không gian
DIRECTIONS = {
    "north": ["phía bắc", "hướng bắc", "mạn bắc", "miền bắc", "đằng bắc", "bên bắc", "góc bắc", "phía bắc bản đồ",
              "phía trên", "bên trên", "mạn trên", "đằng trên", "ở trên", "trên cùng", "phía trên cùng", "đầu trên",
              "nửa trên", "phần trên", "phía trên bản đồ", "bên trên bản đồ", "mé trên", "mé bắc",
              "cạnh trên", "rìa trên", "phía đầu bản đồ"],
    "south": ["phía nam", "hướng nam", "mạn nam", "miền nam", "đằng nam", "bên nam", "góc nam", "phía nam bản đồ",
              "phía dưới", "bên dưới", "mạn dưới", "đằng dưới", "ở dưới", "dưới cùng", "phía dưới cùng", "đầu dưới",
              "nửa dưới", "phần dưới", "phía dưới bản đồ", "bên dưới bản đồ", "mé dưới", "mé nam",
              "cạnh dưới", "rìa dưới", "phía cuối bản đồ"],
    "west": ["phía tây", "hướng tây", "mạn tây", "đằng tây", "bên tây", "góc tây", "phía tây bản đồ", "bên trái",
             "phía trái", "mạn trái", "đằng trái", "bên tay trái", "nửa trái", "phần trái", "trái bản đồ",
             "bên trái bản đồ", "phía trái bản đồ", "mạn trái bản đồ", "mé trái", "mé tây", "cạnh trái", "rìa trái",
             "ngoài cùng bên trái"],
    "east": ["phía đông", "hướng đông", "mạn đông", "đằng đông", "bên đông", "góc đông", "phía đông bản đồ",
             "bên phải", "phía phải", "mạn phải", "đằng phải", "bên tay phải", "nửa phải", "phần phải",
             "phải bản đồ", "bên phải bản đồ", "phía phải bản đồ", "mạn phải bản đồ", "mé phải", "mé đông",
             "cạnh phải", "rìa phải", "ngoài cùng bên phải"],
}
NEAR = ["gần", "sát", "cạnh", "kế bên", "bên cạnh", "gần với", "gần sát", "sát bên", "sát với", "kề", "kề bên",
        "liền kề", "ngay cạnh", "gần kề"]
FAR = ["xa", "cách xa", "xa với", "xa khỏi", "khá xa", "ở xa"]
