"""
Ingest Luật Thương mại 2005 into statutory knowledge base.
"""
import json
from pathlib import Path

CHUNKS = [
    {
        "chunk_id": "VN_LTM_2005_D292",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 292,
        "article_title": "Các loại chế tài trong thương mại",
        "canonical_citation": "Luật Thương mại 2005 - Điều 292",
        "content": (
            "Điều 292. Các loại chế tài trong thương mại\n"
            "1. Buộc thực hiện đúng hợp đồng.\n"
            "2. Phạt vi phạm.\n"
            "3. Buộc bồi thường thiệt hại.\n"
            "4. Tạm ngừng thực hiện hợp đồng.\n"
            "5. Đình chỉ thực hiện hợp đồng.\n"
            "6. Huỷ bỏ hợp đồng.\n"
            "7. Các biện pháp khác do các bên thoả thuận không trái với nguyên tắc cơ bản của pháp luật Việt Nam, điều ước quốc tế mà Cộng hòa xã hội chủ nghĩa Việt Nam là thành viên và tập quán thương mại quốc tế."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D294",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 294,
        "article_title": "Các trường hợp miễn trách nhiệm đối với hành vi vi phạm",
        "canonical_citation": "Luật Thương mại 2005 - Điều 294",
        "content": (
            "Điều 294. Các trường hợp miễn trách nhiệm đối với hành vi vi phạm\n"
            "1. Bên vi phạm hợp đồng được miễn trách nhiệm trong các trường hợp sau đây:\n"
            "a) Xảy ra trường hợp miễn trách nhiệm mà các bên đã thoả thuận;\n"
            "b) Xảy ra sự kiện bất khả kháng;\n"
            "c) Hành vi vi phạm của một bên hoàn toàn do lỗi của bên kia;\n"
            "d) Hành vi vi phạm của một bên do thực hiện quyết định của cơ quan quản lý nhà nước có thẩm quyền mà các bên không thể biết được vào thời điểm giao kết hợp đồng.\n"
            "2. Bên vi phạm hợp đồng có nghĩa vụ chứng minh các trường hợp miễn trách nhiệm."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D300",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 300,
        "article_title": "Phạt vi phạm",
        "canonical_citation": "Luật Thương mại 2005 - Điều 300",
        "content": (
            "Điều 300. Phạt vi phạm\n"
            "Phạt vi phạm là việc bên bị vi phạm yêu cầu bên vi phạm trả một khoản tiền phạt do vi phạm hợp đồng nếu trong hợp đồng có thoả thuận, trừ các trường hợp miễn trách nhiệm quy định tại Điều 294 của Luật này."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D301",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 301,
        "article_title": "Mức phạt vi phạm",
        "canonical_citation": "Luật Thương mại 2005 - Điều 301",
        "content": (
            "Điều 301. Mức phạt vi phạm\n"
            "Mức phạt đối với vi phạm nghĩa vụ hợp đồng hoặc tổng mức phạt đối với nhiều vi phạm do các bên thoả thuận trong hợp đồng, nhưng không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm, trừ trường hợp quy định tại Điều 266 của Luật này."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D302",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 302,
        "article_title": "Bồi thường thiệt hại",
        "canonical_citation": "Luật Thương mại 2005 - Điều 302",
        "content": (
            "Điều 302. Bồi thường thiệt hại\n"
            "1. Bồi thường thiệt hại là việc bên vi phạm bồi thường những tổn thất do hành vi vi phạm hợp đồng gây ra cho bên bị vi phạm.\n"
            "2. Giá trị bồi thường thiệt hại bao gồm giá trị tổn thất thực tế, trực tiếp mà bên bị vi phạm phải chịu do bên vi phạm gây ra và khoản lợi trực tiếp mà bên bị vi phạm đáng lẽ được hưởng nếu không có hành vi vi phạm."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D303",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 303,
        "article_title": "Căn cứ phát sinh trách nhiệm bồi thường thiệt hại",
        "canonical_citation": "Luật Thương mại 2005 - Điều 303",
        "content": (
            "Điều 303. Căn cứ phát sinh trách nhiệm bồi thường thiệt hại\n"
            "Trừ các trường hợp miễn trách nhiệm quy định tại Điều 294 của Luật này, trách nhiệm bồi thường thiệt hại phát sinh khi có đủ các yếu tố sau đây:\n"
            "1. Có hành vi vi phạm hợp đồng;\n"
            "2. Có thiệt hại thực tế;\n"
            "3. Hành vi vi phạm hợp đồng là nguyên nhân trực tiếp gây ra thiệt hại."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D307",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 307,
        "article_title": "Quan hệ giữa chế tài phạt vi phạm và chế tài bồi thường thiệt hại",
        "canonical_citation": "Luật Thương mại 2005 - Điều 307",
        "content": (
            "Điều 307. Quan hệ giữa chế tài phạt vi phạm và chế tài bồi thường thiệt hại\n"
            "1. Trường hợp các bên không có thỏa thuận phạt vi phạm thì bên bị vi phạm chỉ có quyền yêu cầu bồi thường thiệt hại, trừ trường hợp Luật này có quy định khác.\n"
            "2. Trường hợp các bên có thỏa thuận phạt vi phạm thì bên bị vi phạm có quyền áp dụng cả chế tài phạt vi phạm và buộc bồi thường thiệt hại, trừ trường hợp các bên có thoả thuận khác."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D310",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 310,
        "article_title": "Đình chỉ thực hiện hợp đồng",
        "canonical_citation": "Luật Thương mại 2005 - Điều 310",
        "content": (
            "Điều 310. Đình chỉ thực hiện hợp đồng\n"
            "Trừ các trường hợp miễn trách nhiệm quy định tại Điều 294 của Luật này, chế tài đình chỉ thực hiện hợp đồng được áp dụng khi xảy ra hành vi vi phạm mà các bên đã thoả thuận là điều kiện để đình chỉ hoặc một bên vi phạm cơ bản nghĩa vụ hợp đồng."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D311",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 311,
        "article_title": "Hậu quả pháp lý của việc đình chỉ thực hiện hợp đồng",
        "canonical_citation": "Luật Thương mại 2005 - Điều 311",
        "content": (
            "Điều 311. Hậu quả pháp lý của việc đình chỉ thực hiện hợp đồng\n"
            "1. Khi hợp đồng bị đình chỉ thực hiện thì hợp đồng chấm dứt từ thời điểm một bên nhận được thông báo đình chỉ. Các bên không phải tiếp tục thực hiện nghĩa vụ hợp đồng. Bên đã thực hiện nghĩa vụ có quyền yêu cầu bên kia thanh toán hoặc thực hiện nghĩa vụ đối ứng.\n"
            "2. Bên bị vi phạm có quyền yêu cầu bồi thường thiệt hại theo quy định tại các điều từ Điều 302 đến Điều 307 của Luật này."
        ),
        "is_active": True
    },
    {
        "chunk_id": "VN_LTM_2005_D315",
        "law_id": "luat-thuong-mai-2005",
        "law_title": "Luật Thương mại 2005",
        "article_num": 315,
        "article_title": "Thông báo tạm ngừng, đình chỉ hoặc huỷ bỏ hợp đồng",
        "canonical_citation": "Luật Thương mại 2005 - Điều 315",
        "content": (
            "Điều 315. Thông báo tạm ngừng, đình chỉ hoặc huỷ bỏ hợp đồng\n"
            "Bên tạm ngừng thực hiện hợp đồng, đình chỉ thực hiện hợp đồng hoặc huỷ bỏ hợp đồng phải thông báo ngay cho bên kia biết về việc tạm ngừng, đình chỉ hoặc huỷ bỏ. Trong trường hợp không thông báo ngay mà gây thiệt hại cho bên kia thì bên tạm ngừng thực hiện hợp đồng, đình chỉ thực hiện hợp đồng hoặc huỷ bỏ hợp đồng phải bồi thường thiệt hại."
        ),
        "is_active": True
    }
]

def main():
    out_file = Path("data/extracted_laws/luat-thuong-mai-2005_chunks.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(CHUNKS, f, ensure_ascii=False, indent=2)
    print(f"Created {out_file} with {len(CHUNKS)} chunks.")

if __name__ == "__main__":
    main()

