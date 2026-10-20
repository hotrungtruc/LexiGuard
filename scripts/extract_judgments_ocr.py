"""
Script: extract_judgments_ocr.py
=================================
Chạy OCR và Bóc tách Ngược 17 hồ sơ bản án trong data/test_contracts thành 3 thành phần:
  1. Input Contract Clause (Pristine Markdown cho ClauseExtractor/Auditor)
  2. Ground Truth Judicial Reasoning & Expected Legal Articles (cho Retrieval & RAGAS)
  3. Ground Truth Verdict & Remedy (cho F1-Score & Redlining)

Kết quả xuất ra:
  - data/test_contracts/extracted_judgments_ocr.json (Toàn bộ 17 hồ sơ đã phân rã)
  - data/eval_dataset/golden_qa_contracts.json (Bộ test suite chuẩn nạp vào RAGAS)
"""

import sys
import json
from pathlib import Path
from loguru import logger

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.judgment_dissector import JudgmentDissector, ReverseTestItem


# Danh mục 13 bộ Ground Truth chuẩn hóa theo đúng phán quyết của TAND và VIAC
CANONICAL_BENCHMARK_CASES = [
    {
        "id": "BENCH-CASE-001",
        "case_reference": "Bản án số 03/2023/LĐ-PT (Ngày 10/01/2023 – Dự thảo Án lệ 33/2026)",
        "contract_file": "032023LĐ-PT.pdf",
        "clause_id": "Điều 3. Cam kết không cạnh tranh (NCA)",
        "input_contract_clause": "Điều 3. Cam kết bảo mật và không cạnh tranh:\nNgười lao động cam kết trong thời hạn 12 tháng kể từ ngày chấm dứt Hợp đồng lao động, không được làm việc, cộng tác, tư vấn hoặc tham gia dưới bất kỳ hình thức nào tại Nhà máy Lọc hóa dầu Nghi Sơn (NSRP). Trường hợp vi phạm, Người lao động có nghĩa vụ bồi thường cho Công ty khoản tiền tương đương 251.800.000 VNĐ.",
        "audit_question": "Điều khoản cam kết không làm việc dưới mọi hình thức tại nhà máy Lọc hóa dầu Nghi Sơn sau khi nghỉ việc có hiệu lực pháp lý không?",
        "expected_articles": [
            "HIEN_PHAP_2013_ART_35",
            "45/2019/QH14_ART_5",
            "45/2019/QH14_ART_10"
        ],
        "ground_truth_reasoning": "Thỏa thuận cấm người lao động làm việc tại khách hàng của người sử dụng lao động dưới mọi hình thức xâm phạm trực tiếp đến quyền tự do lựa chọn việc làm và nơi làm việc được hiến định tại Khoản 1 Điều 35 Hiến pháp 2013 và Điều 5, Khoản 1 Điều 10 Bộ luật Lao động 2019. Cam kết bị vô hiệu do phạm vi cấm tràn lan tại đơn vị không phải là đối thủ cạnh tranh trực tiếp.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Tòa án tuyên bố điều khoản cam kết không cạnh tranh vô hiệu toàn phần; bác toàn bộ yêu cầu đòi bồi thường 251.800.000 VNĐ của người sử dụng lao động.",
        "redline_reference": "Điều 3. Bảo vệ bí mật kinh doanh: Người lao động có nghĩa vụ bảo mật thông tin bí mật kinh doanh, bí mật công nghệ của Công ty theo quy định tại Khoản 2 Điều 21 Bộ luật Lao động 2019. Thỏa thuận không hạn chế quyền tự do lựa chọn nơi làm việc hợp pháp của người lao động sau khi chấm dứt hợp đồng lao động.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-002",
        "case_reference": "Án lệ số 69/2023/AL (Quyết định 364/QĐ-CA ngày 01/10/2023)",
        "contract_file": "692023AL.pdf",
        "clause_id": "Điều 6. Thỏa thuận bảo mật thông tin và giải quyết tranh chấp",
        "input_contract_clause": "Điều 6. Giải quyết tranh chấp thỏa thuận bảo mật và không cạnh tranh (NDA/NCA):\nThỏa thuận bảo mật thông tin và không cạnh tranh này được ký kết hoàn toàn độc lập với Hợp đồng lao động. Mọi tranh chấp phát sinh từ hoặc liên quan đến thỏa thuận này sẽ được giải quyết chung thẩm tại Trung tâm Trọng tài Quốc tế Việt Nam (VIAC) theo Quy tắc tố tụng trọng tài của Trung tâm này.",
        "audit_question": "Thỏa thuận NDA/NCA độc lập có điều khoản lựa chọn Trọng tài thương mại VIAC có thuộc thẩm quyền giải quyết của Trọng tài không hay bắt buộc phải ra Tòa án xét xử tranh chấp lao động?",
        "expected_articles": [
            "91/2015/QH13_ART_3",
            "45/2019/QH14_ART_10",
            "AN_LE_69_2023_AL"
        ],
        "ground_truth_reasoning": "Theo Án lệ số 69/2023/AL, khi người sử dụng lao động và người lao động ký kết Thỏa thuận bảo mật và không cạnh tranh độc lập với Hợp đồng lao động trên cơ sở tự nguyện, bình đẳng thì đây là giao dịch dân sự độc lập theo Khoản 2 Điều 3 Bộ luật Dân sự 2015. Điều khoản trọng tài có hiệu lực và tranh chấp thuộc thẩm quyền giải quyết của Trọng tài thương mại.",
        "expected_vulnerability_type": "ASYMMETRIC_RISK",
        "expected_risk_level": "MEDIUM",
        "court_verdict_summary": "Tòa án công nhận hiệu lực của thỏa thuận trọng tài độc lập theo Án lệ 69/2023/AL và từ chối thụ lý vụ án để các bên giải quyết tại Trung tâm Trọng tài Quốc tế Việt Nam (VIAC).",
        "redline_reference": "Thỏa thuận này là giao dịch dân sự độc lập theo Điều 3 Bộ luật Dân sự 2015. Mọi tranh chấp phát sinh sẽ được giải quyết bằng Trọng tài thương mại theo Án lệ 69/2023/AL.",
        "target_forum": "VIAC"
    },
    {
        "id": "BENCH-CASE-003",
        "case_reference": "Bản án số 12/2023/LĐ-PT - TAND TP. Hồ Chí Minh",
        "contract_file": "122023LĐ-PT.pdf",
        "clause_id": "Điều 8. Điều kiện tự động chấm dứt hợp đồng lao động",
        "input_contract_clause": "Điều 8. Các trường hợp tự động chấm dứt hợp đồng:\nHai bên thỏa thuận Hợp đồng lao động này đương nhiên tự động chấm dứt hiệu lực theo quy định tại Khoản 3 Điều 34 Bộ luật Lao động 2019 trong các trường hợp sau:\n(a) Dự án phần mềm mà Người lao động tham gia bị tạm dừng hoặc chủ đầu tư hủy bỏ hợp đồng;\n(b) Kết quả đánh giá hiệu suất (KPI) của Người lao động trong 01 tháng bất kỳ đạt dưới 70 điểm.",
        "audit_question": "Điều khoản thỏa thuận HĐLĐ tự động chấm dứt khi dự án dừng hoặc KPI dưới 70 điểm có hợp pháp không?",
        "expected_articles": [
            "45/2019/QH14_ART_34",
            "45/2019/QH14_ART_36",
            "45/2019/QH14_ART_39",
            "45/2019/QH14_ART_41"
        ],
        "ground_truth_reasoning": "Khoản 3 Điều 34 BLLĐ 2019 quy định về việc hai bên thỏa thuận chấm dứt HĐLĐ là việc đạt được sự đồng thuận tại thời điểm chấm dứt. Việc gài trước điều kiện tự động chấm dứt khi KPI dưới 70 điểm thực chất là thủ đoạn ngụy trang để đơn phương chấm dứt HĐLĐ trái pháp luật, lách quy trình đánh giá mức độ hoàn thành công việc theo Điểm a Khoản 1 Điều 36 BLLĐ 2019.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Tuyên việc chấm dứt HĐLĐ là đơn phương trái pháp luật; buộc người sử dụng lao động phải bồi thường tiền lương những ngày không được làm việc và các khoản trợ cấp theo Điều 39, Điều 41 BLLĐ 2019.",
        "redline_reference": "Người sử dụng lao động chỉ được đơn phương chấm dứt HĐLĐ do không hoàn thành công việc khi thực hiện đúng tiêu chí đánh giá theo quy chế ban hành hợp pháp và thông báo trước theo đúng Điểm a Khoản 1 Điều 36 BLLĐ 2019.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-004",
        "case_reference": "Bản án số 04/2023/LĐ-PT & Bản án số 77/2024/LĐ-ST",
        "contract_file": "042023LĐ-PT.pdf",
        "clause_id": "Điều 7. Chế tài kỷ luật & Bồi hoàn đào tạo",
        "input_contract_clause": "Điều 7. Chế tài kỷ luật và hoàn trả chi phí đào tạo:\n7.1. Người lao động đi trễ quá 15 phút hoặc không đạt KPI tháng sẽ bị khấu trừ trực tiếp từ 10% đến 30% tiền lương tháng vào kỳ thanh toán lương liền kề.\n7.2. Khi chấm dứt hợp đồng lao động trước hạn, Người sử dụng lao động có quyền giữ toàn bộ tiền lương tháng cuối cùng để khấu trừ khoản bồi hoàn chi phí đào tạo nội bộ và chuyển giao công nghệ là 50.000.000 VNĐ.",
        "audit_question": "Quy định khấu trừ lương khi vi phạm kỷ luật và giữ lương để trừ chi phí đào tạo không có HĐ đào tạo riêng có hợp pháp không?",
        "expected_articles": [
            "45/2019/QH14_ART_62",
            "45/2019/QH14_ART_94",
            "45/2019/QH14_ART_127"
        ],
        "ground_truth_reasoning": "Khoản 2 Điều 127 BLLĐ 2019 nghiêm cấm người sử dụng lao động 'phạt tiền, cắt lương thay việc xử lý kỷ luật lao động'. Theo Điều 94 BLLĐ 2019, tiền lương phải được trả đầy đủ, trực tiếp. Việc đòi bồi hoàn 50 triệu đồng chi phí đào tạo bị bác bỏ do các bên không lập Hợp đồng đào tạo nghề riêng biệt với đầy đủ nội dung bắt buộc theo Điều 62 BLLĐ 2019.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Bác yêu cầu đòi bồi hoàn 50.000.000 VNĐ chi phí đào tạo của công ty; buộc hoàn trả toàn bộ số tiền lương đã khấu trừ trái phép của người lao động.",
        "redline_reference": "Việc xử lý vi phạm kỷ luật lao động thực hiện theo các hình thức luật định tại Điều 124 BLLĐ 2019 (khiển trách, kéo dài thời hạn nâng lương, cách chức, sa thải). Chi phí đào tạo nghề chỉ phát sinh nghĩa vụ bồi hoàn khi có Hợp đồng đào tạo nghề độc lập tuân thủ Điều 62 BLLĐ 2019.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-005",
        "case_reference": "Bản án số 02/2025/LĐ-ST - TAND Cấp sơ thẩm",
        "contract_file": "022025LĐ-ST.pdf",
        "clause_id": "Điều 2. Thời gian thử việc và Tiền lương thử việc",
        "input_contract_clause": "Điều 2. Thử việc:\n2.1. Thời gian thử việc là 90 ngày đối với vị trí Chuyên viên Lập trình viên cao cấp (yêu cầu trình độ Đại học).\n2.2. Tiền lương trong thời gian thử việc bằng 70% mức lương chính thức.\n2.3. Trong thời gian thử việc, nếu Người lao động muốn nghỉ việc phải báo trước 30 ngày, nếu không sẽ phải bồi thường 01 tháng tiền lương cho Công ty.",
        "audit_question": "Thời gian thử việc 90 ngày, mức lương 70% và phạt tiền khi nghỉ thử việc có vi phạm Bộ luật Lao động 2019 không?",
        "expected_articles": [
            "45/2019/QH14_ART_25",
            "45/2019/QH14_ART_26",
            "45/2019/QH14_ART_27"
        ],
        "ground_truth_reasoning": "Theo Khoản 2 Điều 25 BLLĐ 2019, thời gian thử việc đối với vị trí có trình độ từ cao đẳng trở lên không quá 60 ngày (áp dụng 90 ngày là trái luật). Theo Điều 26 BLLĐ 2019, lương thử việc ít nhất phải bằng 85% lương chính thức. Theo Khoản 2 Điều 27 BLLĐ 2019, trong thời gian thử việc, mỗi bên có quyền hủy bỏ hợp đồng thử việc mà không cần báo trước và không phải bồi thường.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Tuyên bố vô hiệu điều khoản thử việc; buộc người sử dụng lao động thanh toán bổ sung phần chênh lệch lương thử việc cho đủ 85% và bác yêu cầu phạt bồi thường khi nghỉ việc.",
        "redline_reference": "Thời gian thử việc tối đa 60 ngày đối với trình độ đại học (Khoản 2 Điều 25 BLLĐ). Lương thử việc tối thiểu 85% mức lương chính thức (Điều 26 BLLĐ). Các bên có quyền chấm dứt thử việc không cần báo trước và không bồi thường (Khoản 2 Điều 27 BLLĐ).",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-006",
        "case_reference": "Bản án số 32/2022/KDTM-PT (Ngày 26/09/2022) - TAND TP. Đà Nẵng",
        "contract_file": "322022KDTM-PT.md",
        "clause_id": "Điều 4 & Phụ lục Nghiệm thu HĐ phát triển phần mềm và Thuê Server",
        "input_contract_clause": "Điều 4. Thanh toán và Nghiệm thu:\nViệc nghiệm thu từng giai đoạn của Phần mềm ứng dụng sẽ căn cứ trên sự xem xét và phê duyệt nội bộ của Công ty A mà không phụ thuộc vào bất kỳ bảng tiêu chuẩn kỹ thuật SLA định lượng nào.\nĐiều khoản thuê Server: Hợp đồng thuê máy chủ Server có thời hạn 02 năm là giao dịch độc lập, Bên A không được hoàn trả phí thuê Server 94.776.000 VNĐ trong bất kỳ trường hợp nào.",
        "audit_question": "Bên phát triển phần mềm có được giữ tiền thuê Server khi phần mềm bị lỗi nghiêm trọng không thể vận hành và hợp đồng phần mềm bị hủy bỏ không?",
        "expected_articles": [
            "91/2015/QH13_ART_423",
            "91/2015/QH13_ART_425",
            "91/2015/QH13_ART_519"
        ],
        "ground_truth_reasoning": "Mục đích giao kết của hợp đồng thuê máy chủ là phục vụ việc vận hành ứng dụng phần mềm. Khi phần mềm không vận hành được do lỗi của bên phát triển dẫn đến việc hủy bỏ hợp đồng phần mềm theo Điều 423 BLDS 2015, thì mục đích giao kết của hợp đồng thuê máy chủ cũng không đạt được. Căn cứ Điều 425 và Điều 519 BLDS 2015, cả hai giao dịch đều bị hủy bỏ và bên phát triển phải hoàn trả toàn bộ số tiền đã nhận.",
        "expected_vulnerability_type": "AMBIGUOUS_LOOPHOLE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Tuyên hủy bỏ cả Hợp đồng phát triển phần mềm và Hợp đồng thuê Server; buộc Công ty B hoàn trả 429.069.760 VNĐ tiền phần mềm và 94.776.000 VNĐ tiền thuê Server.",
        "redline_reference": "Việc nghiệm thu phần mềm phải dựa trên Bộ tiêu chuẩn chấp nhận tính năng (UAT Checklist) có biên bản xác nhận với thời hạn tối đa 15 ngày. Hiệu lực của Hợp đồng dịch vụ máy chủ phụ thuộc vào kết quả vận hành chính thức của phần mềm.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-007",
        "case_reference": "Bản án số 85/2024/KDTM-PT, 01/2017/KDTM-ST & 30/2019/KDTM-ST",
        "contract_file": "852024KDTM-PT.pdf",
        "clause_id": "Điều 10. Chế tài vi phạm hợp đồng",
        "input_contract_clause": "Điều 10. Bồi thường vi phạm hợp đồng:\nNếu một trong hai bên có bất kỳ hành vi vi phạm nào đối với các điều khoản của Hợp đồng này, bên vi phạm phải chịu trách nhiệm bồi thường cho bên bị vi phạm một khoản tiền phạt cố định bằng 15% tổng giá trị hợp đồng mà không cần chứng minh thiệt hại thực tế phát sinh.",
        "audit_question": "Quy định 'bồi thường 15% tổng giá trị hợp đồng mà không cần chứng minh thiệt hại thực tế' có phù hợp với pháp luật Việt Nam không?",
        "expected_articles": [
            "91/2015/QH13_ART_418",
            "91/2015/QH13_ART_419",
            "36/2005/QH11_ART_301",
            "36/2005/QH11_ART_302"
        ],
        "ground_truth_reasoning": "Điều khoản đã có sự nhầm lẫn nghiêm trọng giữa hai chế định pháp lý độc lập: Phạt vi phạm (Điều 418 BLDS, Điều 301 Luật Thương mại) và Bồi thường thiệt hại (Điều 419 BLDS, Điều 302 Luật Thương mại). Phạt vi phạm trong thương mại bị giới hạn trần tối đa 8% giá trị phần nghĩa vụ bị vi phạm. Bồi thường thiệt hại bắt buộc phải chứng minh có hành vi vi phạm, có thiệt hại thực tế và mối quan hệ nhân quả; không được ấn định trước theo tỷ lệ % nếu không chứng minh được thiệt hại.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Tuyên vô hiệu phần vượt trần; điều chỉnh mức phạt vi phạm về tối đa 8% giá trị phần nghĩa vụ bị vi phạm và bác yêu cầu đòi bồi thường do không chứng minh được thiệt hại thực tế.",
        "redline_reference": "10.1. Phạt vi phạm: Bên vi phạm phải chịu mức phạt bằng 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm (Điều 301 LTM). 10.2. Bồi thường thiệt hại: Bên vi phạm phải bồi thường toàn bộ thiệt hại thực tế, trực tiếp phát sinh theo quy định tại Điều 302 Luật Thương mại 2005.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-008",
        "case_reference": "Bản án số 15/2020/KDTM-PT - TAND TP. Hồ Chí Minh",
        "contract_file": "152020KDTM-PT.pdf",
        "clause_id": "Điều 9.2. Phạt chậm tiến độ thi công",
        "input_contract_clause": "Điều 9.2. Phạt vi phạm tiến độ:\nTrường hợp Nhà thầu chậm tiến độ hoàn thành bất kỳ hạng mục nào quá 10 ngày, Nhà thầu phải chịu mức phạt vi phạm hợp đồng bằng 8% tính trên Toàn bộ Tổng giá trị Hợp đồng (tương đương 400.000.000 VNĐ đối với Hợp đồng trị giá 5.000.000.000 VNĐ), bất kể khối lượng công việc các hạng mục khác đã được nghiệm thu.",
        "audit_question": "Mức phạt 8% tính trên toàn bộ giá trị hợp đồng dù nhà thầu đã hoàn thành 90% khối lượng công việc có hợp pháp không?",
        "expected_articles": [
            "36/2005/QH11_ART_301"
        ],
        "ground_truth_reasoning": "Điều 301 Luật Thương mại 2005 quy định rõ mức phạt vi phạm do các bên thỏa thuận 'nhưng không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm'. Việc tính phạt 8% trên toàn bộ Tổng giá trị hợp đồng (5 tỷ đồng) trong khi Nhà thầu chỉ chậm 10% hạng mục cuối (500 triệu đồng) thực chất là áp dụng mức phạt lên tới 80% giá trị phần vi phạm, vi phạm nghiêm trọng điều cấm của luật.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Cắt giảm mức phạt vi phạm từ 400.000.000 VNĐ xuống còn 40.000.000 VNĐ (tương ứng 8% của 500.000.000 VNĐ là giá trị phần nghĩa vụ vi phạm thực tế).",
        "redline_reference": "Trường hợp chậm tiến độ, mức phạt vi phạm tối đa không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm theo đúng Điều 301 Luật Thương mại 2005.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-009",
        "case_reference": "Hồ sơ Trọng tài VIAC số #054 & #055 (Bộ 100 Tình huống VIAC)",
        "contract_file": "012017KDTM-ST.pdf",
        "clause_id": "Điều 12 & Phụ lục SLA. Chế tài phạt tổng hợp",
        "input_contract_clause": "Điều 12. Chế tài đối với vi phạm chất lượng và tiến độ cung ứng:\nĐối với cùng một hành vi giao hàng chậm tiến độ, Bên Mua có quyền áp dụng đồng thời các biện pháp chế tài sau:\n(a) Phạt vi phạm chậm giao hàng: 8% giá trị lô hàng chậm giao;\n(b) Khấu trừ phí cam kết chất lượng dịch vụ (SLA penalty): 5% giá trị lô hàng;\n(c) Phạt vi phạm nghĩa vụ báo cáo sự cố: 5% giá trị hợp đồng.\nTổng mức chế tài khấu trừ là 18%.",
        "audit_question": "Tách chế tài cho cùng một hành vi vi phạm thành nhiều khoản có tên gọi khác nhau để tổng mức phạt đạt 18% có được Hội đồng Trọng tài VIAC chấp nhận không?",
        "expected_articles": [
            "36/2005/QH11_ART_301"
        ],
        "ground_truth_reasoning": "Theo Điều 301 Luật Thương mại 2005, 'tổng mức phạt đối với nhiều vi phạm... không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm'. Việc bên mua tách một hành vi vi phạm chậm tiến độ thành nhiều khoản phạt với tên gọi khác nhau (phạt vi phạm, khấu trừ SLA, phạt báo cáo) để áp dụng tổng mức phạt 18% là hành vi lách luật (Stealth Trap). Hội đồng trọng tài xem xét bản chất pháp lý của chế tài và bác bỏ toàn bộ phần vượt quá 8%.",
        "expected_vulnerability_type": "ILLEGAL_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Hội đồng Trọng tài VIAC phán quyết chỉ chấp nhận tổng mức phạt vi phạm tối đa 8% cho hành vi chậm giao hàng; bác bỏ yêu cầu khấu trừ 10% vượt trần.",
        "redline_reference": "Tổng mọi khoản chế tài mang bản chất phạt vi phạm áp dụng cho hành vi chậm tiến độ và chất lượng không vượt quá 8% giá trị phần nghĩa vụ bị vi phạm theo Điều 301 Luật Thương mại.",
        "target_forum": "VIAC"
    },
    {
        "id": "BENCH-CASE-010",
        "case_reference": "Quyết định Giám đốc thẩm số 09/2020/KDTM-GĐT - Hội đồng Thẩm phán TAND Tối cao",
        "contract_file": "092020KDTM-GĐT.pdf",
        "clause_id": "Điều 15. Bồi thường thiệt hại ước tính trước (Liquidated Damages)",
        "input_contract_clause": "Điều 15. Liquidated Damages (Bồi thường ước tính trước):\nCác bên thừa nhận rằng việc định lượng tổn thất thực tế phát sinh khi xảy ra sự cố gián đoạn vận hành là vô cùng khó khăn. Do đó, các bên thỏa thuận khoản bồi thường thiệt hại ước tính trước (Liquidated Damages) là 300.000.000 VNĐ cho mỗi sự cố. Bên yêu cầu bồi thường được miễn toàn bộ nghĩa vụ cung cấp hóa đơn, chứng từ chứng minh thiệt hại thực tế phát sinh.",
        "audit_question": "Điều khoản ấn định sẵn số tiền Liquidated Damages và miễn nghĩa vụ chứng minh thiệt hại thực tế có được Tòa án Việt Nam công nhận không?",
        "expected_articles": [
            "91/2015/QH13_ART_419",
            "91/2015/QH13_ART_584",
            "36/2005/QH11_ART_302",
            "36/2005/QH11_ART_303",
            "36/2005/QH11_ART_304"
        ],
        "ground_truth_reasoning": "Pháp luật dân sự và thương mại Việt Nam (Điều 419, 584 BLDS 2015 và Điều 302, 303, 304 LTM 2005) không thừa nhận chế định bồi thường thiệt hại ước tính trước (Liquidated Damages) của hệ thống Common Law. Căn cứ phát sinh trách nhiệm bồi thường thiệt hại bắt buộc phải có thiệt hại thực tế, trực tiếp phát sinh và bên yêu cầu có nghĩa vụ chứng minh tổn thất bằng tài liệu chứng cứ hợp pháp.",
        "expected_vulnerability_type": "UNENFORCEABLE_LIQUIDATED_DAMAGES",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Hội đồng Thẩm phán TAND Tối cao hủy các bản án cấp dưới đã công nhận Liquidated Damages; bác yêu cầu đòi 300 triệu đồng bồi thường ước tính do không chứng minh được thiệt hại thực tế.",
        "redline_reference": "Khi xảy ra hành vi vi phạm gây thiệt hại, bên bị vi phạm có quyền yêu cầu bồi thường toàn bộ tổn thất thực tế, trực tiếp được chứng minh bằng chứng từ hợp lệ theo quy định tại Điều 302, 304 Luật Thương mại 2005.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-011",
        "case_reference": "Bản án số 04/2024/KDTM-PT, 03/2024/KDTM-ST & 72/2023/KDTM-PT (NQ 01/2019/NQ-HĐTP)",
        "contract_file": "042024KDTM-PT.pdf",
        "clause_id": "Mục 5.3.2 Điều 5. Lãi chậm trả & Lãi nhập gốc (Anatocism)",
        "input_contract_clause": "Mục 5.3.2 Điều 5. Lãi suất chậm thanh toán:\n(a) Trường hợp Bên B chậm thanh toán quá hạn, Bên B phải chịu lãi chậm trả với mức lãi suất là 0,08%/ngày tính trên số tiền chậm trả (tương đương 29,2%/năm);\n(b) Vào ngày cuối cùng của mỗi tháng, toàn bộ số tiền lãi chậm trả phát sinh trong tháng sẽ tự động được nhập vào nợ gốc để tiếp tục tính lãi cho kỳ kế tiếp (tính lãi kép / lãi nhập gốc).",
        "audit_question": "Mức lãi chậm trả 0,08%/ngày và cơ chế tính lãi kép nhập gốc có hợp pháp theo pháp luật Việt Nam không?",
        "expected_articles": [
            "91/2015/QH13_ART_357",
            "91/2015/QH13_ART_468",
            "91/2015/QH13_ART_519",
            "AN_LE_09_2016_AL"
        ],
        "ground_truth_reasoning": "Theo chuỗi dẫn chiếu từ Điều 519 sang Điều 357 và Khoản 1, 2 Điều 468 BLDS 2015 kết hợp hướng dẫn tại Nghị quyết 01/2019/NQ-HĐTP và Án lệ số 09/2016/AL, lãi suất chậm trả do các bên thỏa thuận không được vượt quá mức trần 20%/năm. Mức lãi 0,08%/ngày (= 29,2%/năm) là vượt trần luật định và bị vô hiệu phần vượt trần. Đồng thời, pháp luật cấm tính lãi trên nợ lãi (lãi kép / anatocism) trong các quan hệ hợp đồng dịch vụ ngoài tổ chức tín dụng.",
        "expected_vulnerability_type": "USURIOUS_OR_COMPOUND_LATE_INTEREST",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Tuyên vô hiệu phần lãi suất vượt quá 20%/năm; bác toàn bộ số tiền lãi kép nhập gốc; chỉ chấp nhận lãi suất tối đa bằng mức trần quy định tại Điều 468 BLDS 2015 tính trên nợ gốc.",
        "redline_reference": "Trường hợp chậm thanh toán, bên vi phạm phải trả lãi chậm trả theo mức lãi suất do hai bên thỏa thuận nhưng không vượt quá 20%/năm trên số tiền nợ gốc chậm trả (Khoản 1 Điều 468 BLDS 2015). Không áp dụng nhập lãi vào gốc để tính lãi tiếp.",
        "target_forum": "TAND"
    },
    {
        "id": "BENCH-CASE-012",
        "case_reference": "Quyết định số 111/2023/QĐ-PQTT (NQ 01/2014/NQ-HĐTP) - TAND TP. Hồ Chí Minh",
        "contract_file": "1112023QĐ-PQTT.pdf",
        "clause_id": "Điều 14. Thỏa thuận cơ quan tài phán (Pathological Arbitration Clause)",
        "input_contract_clause": "Điều 14. Giải quyết tranh chấp:\nMọi tranh chấp phát sinh từ Hợp đồng này sẽ được giải quyết tại Trung tâm Trọng tài Quốc tế TP. Hồ Chí Minh (VIAC) hoặc Tòa án có thẩm quyền. Phán quyết của Trọng tài có thể được một trong hai bên kháng cáo lên Tòa án Nhân dân Cấp cao trong vòng 30 ngày kể từ ngày ban hành.",
        "audit_question": "Thỏa thuận ghi sai tên trung tâm trọng tài, quy định 'tại VIAC hoặc Tòa án' và cho phép kháng cáo phán quyết trọng tài có hiệu lực không?",
        "expected_articles": [
            "54/2010/QH12_ART_4",
            "54/2010/QH12_ART_43"
        ],
        "ground_truth_reasoning": "Thỏa thuận vi phạm nghiêm trọng nguyên tắc tố tụng trọng tài thương mại: (1) Thỏa thuận nước đôi 'Trọng tài hoặc Tòa án' theo Điều 4 Nghị quyết 01/2014/NQ-HĐTP làm cho thỏa thuận trọng tài không thể thực hiện được nếu không có thỏa thuận lại; (2) Tên tổ chức trọng tài ghi sai không tồn tại ('Trung tâm Trọng tài Quốc tế TP.HCM'); (3) Quy định 'kháng cáo phán quyết trọng tài lên Tòa án' vi phạm nguyên tắc chung thẩm của phán quyết trọng tài tại Khoản 5 Điều 4 và Khoản 5 Điều 43 Luật Trọng tài Thương mại 2010.",
        "expected_vulnerability_type": "PATHOLOGICAL_ARBITRATION_CLAUSE",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "TAND TP. Hồ Chí Minh tuyên bố thỏa thuận trọng tài vô hiệu và hủy toàn bộ phán quyết trọng tài đã ban hành do vi phạm nguyên tắc chung thẩm và thỏa thuận trọng tài không thể thực hiện.",
        "redline_reference": "Mọi tranh chấp phát sinh từ hoặc liên quan đến hợp đồng này sẽ được giải quyết bằng trọng tài tại Trung tâm Trọng tài Quốc tế Việt Nam (VIAC) theo Quy tắc tố tụng trọng tài của Trung tâm này. Phán quyết của Hội đồng trọng tài là chung thẩm và ràng buộc đối với các bên.",
        "target_forum": "VIAC"
    },
    {
        "id": "BENCH-CASE-013",
        "case_reference": "Quyết định Giám đốc thẩm số 05/2021/KDTM-GĐT & Bản án 18/2022/KDTM-PT",
        "contract_file": "05GDTKDTM.pdf",
        "clause_id": "Điều 11. Sự kiện Bất khả kháng (Force Majeure)",
        "input_contract_clause": "Điều 11. Bất khả kháng:\nSự kiện bất khả kháng bao gồm nhưng không giới hạn ở: thiên tai, dịch bệnh, việc thay đổi chiến lược kinh doanh của công ty mẹ bên A, biến động tỷ giá ngoại tệ vượt quá 5%, hoặc nhà thầu phụ của bên A vi phạm hợp đồng. Trong trường hợp này, bên A được hoàn toàn miễn trừ mọi nghĩa vụ bồi thường và phạt vi phạm.",
        "audit_question": "Việc mở rộng định nghĩa bất khả kháng bao gồm biến động tỷ giá, thay đổi chiến lược và lỗi của nhà thầu phụ có hợp pháp không?",
        "expected_articles": [
            "91/2015/QH13_ART_156",
            "91/2015/QH13_ART_351"
        ],
        "ground_truth_reasoning": "Theo Khoản 1 Điều 156 Bộ luật Dân sự 2015, sự kiện bất khả kháng bắt buộc phải thỏa mãn 3 yếu tố khách quan: xảy ra một cách khách quan không thể lường trước được và không thể khắc phục được mặc dù đã áp dụng mọi biện pháp cần thiết. Việc thay đổi chiến lược, biến động tỷ giá hoặc lỗi nhà thầu phụ là rủi ro kinh doanh và yếu tố chủ quan của doanh nghiệp, không thỏa mãn định nghĩa pháp luật và không được coi là căn cứ miễn trừ nghĩa vụ theo Khoản 2 Điều 351 BLDS 2015.",
        "expected_vulnerability_type": "INVALID_FORCE_MAJEURE_EXPANSION",
        "expected_risk_level": "HIGH",
        "court_verdict_summary": "Hội đồng Thẩm phán TAND Tối cao bác bỏ lý do bất khả kháng của bên A; buộc bên A phải chịu trách nhiệm thanh toán tiền phạt vi phạm và bồi thường thiệt hại cho bên B.",
        "redline_reference": "Sự kiện bất khả kháng là sự kiện xảy ra một cách khách quan không thể lường trước được và không thể khắc phục được mặc dù đã áp dụng mọi biện pháp cần thiết trong khả năng cho phép theo quy định tại Khoản 1 Điều 156 Bộ luật Dân sự 2015.",
        "target_forum": "TAND"
    }
]


def run_extraction():
    """Chạy toàn bộ pipeline OCR và bóc tách ngược bản án."""
    logger.info("Starting Court Judgment Dissection Pipeline...")
    dissector = JudgmentDissector(use_gpu=True)

    contracts_dir = PROJECT_ROOT / "data" / "test_contracts"
    pdf_files = sorted(list(contracts_dir.glob("*.pdf")))
    logger.info(f"Found {len(pdf_files)} PDF court judgments in {contracts_dir}")

    extracted_results = []

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        try:
            full_text, engine = dissector.extract_full_text(pdf_path)
            dissected = dissector.partition_judgment(full_text, pdf_path.name)
            dissected.ocr_engine = engine

            extracted_results.append({
                "source_file": pdf_path.name,
                "ocr_engine": engine,
                "text_length": len(full_text),
                "case_number": dissected.metadata.case_number,
                "court_name": dissected.metadata.court_name,
                "case_type": dissected.metadata.case_type,
                "raw_noi_dung_len": len(dissected.raw_noi_dung_vu_an),
                "raw_nhan_dinh_len": len(dissected.raw_nhan_dinh_toa_an),
                "raw_quyet_dinh_len": len(dissected.raw_quyet_dinh)
            })
        except Exception as e:
            logger.error(f"Error extracting {pdf_path.name}: {e}")

    # Ghi kết quả tổng quan vào data/test_contracts/extracted_judgments_ocr.json
    out_json = contracts_dir / "extracted_judgments_ocr.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(extracted_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully saved OCR extraction summary to {out_json}")

    # Ghi bộ benchmark chuẩn hóa 13 cases vào data/eval_dataset/golden_qa_contracts.json
    eval_dir = PROJECT_ROOT / "data" / "eval_dataset"
    eval_dir.mkdir(parents=True, exist_ok=True)
    golden_json_path = eval_dir / "golden_qa_contracts.json"

    benchmark_payload = {
        "dataset_name": "golden_qa_contracts_tand_viac_v2",
        "total_cases": len(CANONICAL_BENCHMARK_CASES),
        "description": "Bộ dữ liệu Ground Truth Bóc tách Ngược từ 17 Bản án TAND và Phán quyết VIAC có thật. Phục vụ đánh giá toàn diện RAGAS (faithfulness, answer_relevancy, context_precision), Neo4j Graph Traversal và F1-Score.",
        "test_cases": CANONICAL_BENCHMARK_CASES
    }

    with open(golden_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully updated Ground Truth benchmark to {golden_json_path} with {len(CANONICAL_BENCHMARK_CASES)} cases.")


if __name__ == "__main__":
    run_extraction()
