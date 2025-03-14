"""Prompt templates and manager."""
from typing import Dict, Any
from src.core.llm.interfaces import PromptTemplateInterface

PROMPT_TEMPLATES = {
    "query_decomposition": {
        "en": """You are an expert at breaking down complex queries into simpler sub-queries.
Given a query, create 3-5 sub-queries that will help address the main question comprehensively.

Consider:
- Key concepts and definitions
- Related topics and themes
- Supporting evidence needed
- Potential counterpoints

Main query: {query}

Generate sub-queries that will help build a complete answer.""",

        "vi": """Bạn là một chuyên gia trong việc phân tích các câu hỏi phức tạp thành các câu hỏi đơn giản hơn.
Với câu hỏi được cung cấp, hãy tạo CHÍNH XÁC 5 câu hỏi phụ sẽ giúp trả lời câu hỏi chính một cách toàn diện.

Cần xem xét:
- Các khái niệm và định nghĩa chính
- Các chủ đề và chủ điểm liên quan
- Bằng chứng cần thiết
- Các quan điểm đối lập tiềm năng

Câu hỏi chính: {query}

Trả về kết quả theo định dạng JSON như sau:
{{
    "sub_queries": [
        {{
            "id": 1,
            "text": "Câu hỏi phụ 1?"
        }},
        {{
            "id": 2,
            "text": "Câu hỏi phụ 2?"
        }},
        ...
    ]
}}

Lưu ý:
- Phải trả về ĐÚNG 5 câu hỏi phụ
- Mỗi câu hỏi phải kết thúc bằng dấu hỏi (?)
- Không thêm giải thích hay chú thích
- Chỉ trả về JSON, không thêm text khác"""
    },

    "summarization": {
        "en": """Create a clear and concise summary of the following text while preserving key information and context.
Focus on:
- Main ideas and concepts
- Important details and examples
- Relationships between ideas
- Section numbers and references

Text to summarize:
{text}

Create a natural summary without any formatting markers or prefixes. The summary should flow as regular text.""",

        "vi": """Tạo bản tóm tắt rõ ràng và súc tích từ văn bản sau đây, giữ được thông tin quan trọng và bối cảnh.
Tập trung vào:
- Ý tưởng và khái niệm chính
- Chi tiết và ví dụ quan trọng
- Mối quan hệ giữa các ý tưởng
- Số mục và tham chiếu

Văn bản cần tóm tắt:
{text}

Tạo bản tóm tắt tự nhiên không có ký hiệu định dạng hoặc tiền tố. Bản tóm tắt nên chảy như văn bản thông thường."""
    },

    "citation_extraction": {
        "en": """Extract citations and references from the following text.
For each citation, identify:
- The exact section or part number
- The relevant text being referenced
- The context of the citation

Text:
{text}

List all citations with their details.""",

        "vi": """Trích xuất trích dẫn và tham chiếu từ văn bản sau.
Với mỗi trích dẫn, xác định:
- Số mục hoặc phần chính xác
- Văn bản được tham chiếu
- Bối cảnh của trích dẫn

Văn bản:
{text}

Liệt kê tất cả trích dẫn với chi tiết của chúng."""
    },

    "answer_synthesis": {
        "en": """Create a comprehensive answer using the provided sub-query results and citations.
The answer should:
- Address the original query directly
- Integrate information from all sub-queries
- Support claims with specific citations
- Maintain a logical flow

Original query: {query}
Sub-query results:
{sub_query_results}
Available citations:
{citations}

Provide a well-structured answer with proper citation references.""",

        "vi": """Tạo câu trả lời toàn diện sử dụng kết quả từ các câu hỏi phụ và trích dẫn được cung cấp.
Câu trả lời cần:
- Trả lời trực tiếp câu hỏi gốc
- Tích hợp thông tin từ tất cả câu hỏi phụ
- Hỗ trợ các luận điểm bằng trích dẫn cụ thể
- Duy trì luồng logic

Câu hỏi gốc: {query}
Kết quả câu hỏi phụ:
{sub_query_results}
Trích dẫn có sẵn:
{citations}

Cung cấp câu trả lời có cấu trúc tốt với các tham chiếu trích dẫn phù hợp."""
    },

    "section_detection": {
        "en": """Analyze this text to identify section information.
        
        Look for:
        1. Article followed by a number
           Example: "Article 5. Rights and Obligations"
        
        2. Chapter followed by a number
           Example: "Chapter I. General Provisions"
        
        3. Section followed by a number
           Example: "Section 1. Scope"
        
        4. Numbered topics or lists
           Example: "1. Rights of professional soldiers:"
        
        Text to analyze:
        {text}
        
        Return:
        {{
            "is_section": true/false,
            "type": "article/chapter/section/topic",
            "number": "section number",
            "title": "section title"
        }}""",
        
        "vi": """Phân tích văn bản này để xác định thông tin về mục.
        
        Tìm kiếm:
        1. Điều (Article) theo sau là số
           Ví dụ: "Điều 5. Quyền và nghĩa vụ"
        
        2. Chương (Chapter) theo sau là số
           Ví dụ: "Chương I. Quy định chung"
        
        3. Mục (Section) theo sau là số
           Ví dụ: "Mục 1. Phạm vi điều chỉnh"
        
        4. Chủ đề hoặc danh sách được đánh số
           Ví dụ: "1. Quyền của quân nhân chuyên nghiệp:"
        
        Văn bản cần phân tích:
        {text}
        
        Trả về:
        {{
            "is_section": true/false,
            "type": "article/chapter/section/topic",
            "number": "số mục",
            "title": "tiêu đề mục"
        }}"""
    },

    "intermediate_summarization": {
        "en": """Create a natural and comprehensive summary from multiple chunk summaries.
Focus on:
- Synthesizing information across chunks
- Identifying common themes and key points
- Maintaining relationships between concepts
- Creating a coherent narrative
- Preserving important details and context

Content to summarize:
{text}

Create a clear summary that flows naturally, without adding any prefixes like 'Summary:' or 'Tóm tắt:'.""",

        "vi": """Tạo bản tóm tắt tự nhiên và toàn diện từ nhiều bản tóm tắt đoạn.
Tập trung vào:
- Tổng hợp thông tin từ các đoạn
- Xác định chủ đề chung và điểm chính
- Duy trì mối quan hệ giữa các khái niệm
- Tạo tường thuật mạch lạc
- Bảo toàn chi tiết và ngữ cảnh quan trọng

Nội dung cần tóm tắt:
{text}

Tạo bản tóm tắt rõ ràng, chảy tự nhiên, không thêm các tiền tố như 'Summary:' hoặc 'Tóm tắt:'."""
    },

    "document_summarization": {
        "en": """Create a clear and comprehensive final summary of the document.
Focus on:
- Main themes and key points across the document
- Critical relationships and dependencies
- Important findings and conclusions
- Logical structure and flow
- Key concepts for query analysis and answer synthesis

Content to summarize:
{text}

Create a natural summary that flows smoothly, without adding any prefixes like 'Summary:' or 'Tóm tắt:'.""",

        "vi": """Tạo bản tóm tắt cuối cùng rõ ràng và toàn diện cho tài liệu.
Tập trung vào:
- Chủ đề chính và điểm quan trọng trong toàn bộ tài liệu
- Mối quan hệ và phụ thuộc quan trọng
- Phát hiện và kết luận quan trọng
- Cấu trúc và luồng logic
- Khái niệm chính cho phân tích truy vấn và tổng hợp câu trả lời

Nội dung cần tóm tắt:
{text}

Tạo bản tóm tắt tự nhiên, chảy mượt mà, không thêm các tiền tố như 'Summary:' hoặc 'Tóm tắt:'."""
    }
}

class PromptManager(PromptTemplateInterface):
    """Manager for prompt templates."""
    
    def __init__(self):
        self.templates = PROMPT_TEMPLATES
    
    def get_prompt(self, prompt_type: str, language: str = "vi") -> str:
        """Get prompt template for given type and language."""
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        if language not in ["en", "vi"]:
            raise ValueError(f"Unsupported language: {language}")
            
        return self.templates[prompt_type][language]
    
    def format_prompt(
        self,
        prompt_type: str,
        language: str = "vi",
        **kwargs: Any
    ) -> str:
        """Format prompt template with variables."""
        template = self.get_prompt(prompt_type, language)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {str(e)}")