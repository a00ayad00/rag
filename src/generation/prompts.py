"""
Bilingual Prompt Engineering and Refusal Templates.
Enforces strict factual adherence, explicit source citations, and exact language matching.
"""

import re
from typing import List, Tuple
from src.retrieval.vector_store import RetrievedPassage

REFUSAL_ENGLISH = "I apologize, but the provided documents do not contain sufficient information to answer this question."
REFUSAL_ARABIC = "عذراً، الوثائق المرفقة لا تحتوي على معلومات كافية للإجابة على هذا السؤال."

SYSTEM_PROMPT_ENGLISH = """You are a highly precise, factual document Q&A assistant.
You strictly answer questions using ONLY the provided context passages.

RULES:
1. LANGUAGE: You MUST respond in the EXACT language of the user's question (English for English queries, Arabic for Arabic queries).
2. CITATIONS: You MUST cite the source of every fact in your response using the exact format: [filename, Page X].
3. STRICT REFUSAL: If the provided context passages do not contain enough facts to directly answer the question, you MUST output EXACTLY:
"{refusal_en}"
Do not guess, do not speculate, and do not use outside knowledge.
4. TONE: Concise, direct, professional, and completely objective.
"""

SYSTEM_PROMPT_ARABIC = """أنت مساعد بحثي دقيق للإجابة على الأسئلة المستندة إلى الوثائق والتقارير.
يجب عليك الالتزام الصارم بالإجابة فقط وحصراً من واقع النصوص والسياقات المرفقة أدناه.

القواعد الصارمة:
1. لغة الإجابة: يجب أن تجيب بنفس لغة السؤال تماماً (إذا كان السؤال باللغة العربية يجب أن تكون الإجابة باللغة العربية بالكامل).
2. التوثيق والمصادر: يجب توثيق كل حقيقة أو رقم بذكر المصدر ورقم الصفحة بالصيغة: [اسم_الملف, Page X].
3. رفض الإجابة عند عدم وجود المعلومة: إذا كانت النصوص المرفقة لا تحتوي على إجابة كافية ومؤكدة للسؤال، يجب عليك الرد حصراً بهذه العبارة دون زيادة أو نقصان:
"{refusal_ar}"
يُمنع منعاً باتاً التخمين أو الاستعانة بأي معلومات من خارج السياق المرفق.
4. الأسلوب: مباشر، دقيق، موضوعي، وموثق.
"""


class BilingualPromptBuilder:
    """Constructs prompts tailored to detected language and retrieval context."""

    ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Determines if query is Arabic or English."""
        arabic_count = len(cls.ARABIC_CHAR_PATTERN.findall(text))
        return "arabic" if arabic_count >= 2 else "english"

    @classmethod
    def format_context(cls, passages: List[RetrievedPassage]) -> str:
        """Formats retrieved passages into a structured context string with citations."""
        if not passages:
            return "No passages retrieved."

        formatted_blocks = []
        for idx, p in enumerate(passages, start=1):
            block = (
                f"--- [Passage {idx}] ---\n"
                f"Source: {p.filename}\n"
                f"Page: {p.page_number}\n"
                f"Language: {p.language}\n"
                f"Content:\n{p.text.strip()}\n"
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    @classmethod
    def build_prompt_messages(cls, question: str, passages: List[RetrievedPassage]) -> Tuple[str, str, str]:
        """
        Builds (system_prompt, user_prompt, language).
        """
        language = cls.detect_language(question)
        context_str = cls.format_context(passages)

        if language == "arabic":
            system_prompt = SYSTEM_PROMPT_ARABIC.format(refusal_ar=REFUSAL_ARABIC)
            user_prompt = (
                f"السياق المرجعي:\n{context_str}\n\n"
                f"السؤال:\n{question}\n\n"
                f"الإجابة الموثقة (باللغة العربية مع ذكر المصادر [filename, Page X]):"
            )
        else:
            system_prompt = SYSTEM_PROMPT_ENGLISH.format(refusal_en=REFUSAL_ENGLISH)
            user_prompt = (
                f"Reference Context:\n{context_str}\n\n"
                f"Question:\n{question}\n\n"
                f"Factual Answer (in English with source citations [filename, Page X]):"
            )

        return system_prompt, user_prompt, language

    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        """Extracts cited documents from answer text using regex."""
        # Matches patterns like [Report.pdf, Page 2] or [EN_Report_01.pdf, Page 1]
        pattern = re.compile(r"\[([A-Za-z0-9_\-\.\u0600-\u06FF\s]+,\s*(?:Page|صفحة)\s*\d+)\]")
        matches = pattern.findall(text)
        return list(dict.fromkeys(matches))  # deduplicate preserving order
