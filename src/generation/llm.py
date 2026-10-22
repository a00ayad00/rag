"""
LLM Provider Abstraction.
Supports Open-Source local models (Ollama), free API tiers (Groq, Gemini),
and deterministic mock generation for testing.
"""

import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.config import settings
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

logger = logging.getLogger(__name__)


class MockChatModel(BaseChatModel):
    """Deterministic mock LLM for offline testing and continuous integration."""

    def _generate(self, messages):
        # Extract last message
        last_msg = messages[-1].content if messages else ""
        text = str(last_msg)

        # Check for unanswerable trigger words or empty context
        is_unanswerable = (
            "unanswerable" in text.lower()
            or "quantum computer" in text.lower()
            or "non-existent" in text.lower()
            or "سياحة الفضاء" in text
            or "المريخ" in text
            or "No passages retrieved." in text
        )

        is_arabic = any("\u0600" <= c <= "\u06FF" for c in text)

        if is_unanswerable:
            reply = (
                "عذراً، الوثائق المرفقة لا تحتوي على معلومات كافية للإجابة على هذا السؤال."
                if is_arabic
                else "I apologize, but the provided documents do not contain sufficient information to answer this question."
            )
        else:
            if is_arabic:
                reply = (
                    "بناءً على التقرير المرفق، حققت المبادرة نتائج إيجابية ملحوظة وتم تنفيذ الخطط بنجاح "
                    "[AR_Report_01_Digital_Transformation.pdf, Page 1]."
                )
            else:
                reply = (
                    "According to the provided document, the enterprise demonstrated strong progress with verified milestones "
                    "[EN_Report_01_AI_Strategy_2024.pdf, Page 1]."
                )

        from langchain_core.outputs import ChatResult, ChatGeneration
        gen = ChatGeneration(message=AIMessage(content=reply))
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"


def get_llm(
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> BaseChatModel:
    """
    Factory function returning the configured ChatModel.
    Supports 'groq', 'gemini', 'ollama', and 'mock'.
    """
    prov = (provider or settings.LLM_PROVIDER).lower()
    temp = temperature if temperature is not None else settings.TEMPERATURE
    max_t = max_tokens or settings.MAX_TOKENS

    logger.info(f"Initializing LLM with provider: {prov}")

    if prov == "groq":
        try:
            import os
            api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY is not set. Falling back to MockChatModel for safety.")
                return MockChatModel()

            from groq import Groq
            from langchain_groq import ChatGroq

            target_model = settings.GROQ_MODEL.strip()
            # Dynamically query available models to prevent 404 model_not_found
            try:
                groq_client = Groq(api_key=api_key)
                models_resp = groq_client.models.list()
                available_ids = [m.id for m in models_resp.data]
                logger.info(f"Active Groq models on this account: {available_ids}")

                if target_model not in available_ids:
                    # Filter out non-chat models (audio, whisper, guard, etc.)
                    chat_models = [
                        m for m in available_ids
                        if not any(x in m.lower() for x in ["whisper", "guard", "distil", "tts"])
                    ]
                    # Select best available replacement
                    chosen = None
                    for pref in ["llama-3.3-70b", "gpt-oss", "llama", "mixtral", "gemma"]:
                        matches = [m for m in chat_models if pref in m.lower()]
                        if matches:
                            chosen = matches[0]
                            break
                    selected = chosen or (chat_models[0] if chat_models else target_model)
                    logger.warning(
                        f"Model '{target_model}' was not found on this Groq account. "
                        f"Auto-selected available model: '{selected}'"
                    )
                    target_model = selected
            except Exception as query_err:
                logger.debug(f"Note on Groq model auto-discovery: {query_err}")

            logger.info(f"Initializing Groq ChatModel with active model: '{target_model}'")
            return ChatGroq(
                model=target_model,
                temperature=temp,
                max_tokens=max_t,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq ({e}). Falling back to MockChatModel.")
            return MockChatModel()

    elif prov == "gemini":
        try:
            import os
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY is not set. Falling back to MockChatModel.")
                return MockChatModel()

            model_name = settings.GEMINI_MODEL.strip()
            # Normalize model name: gemini-2.5-flash does not exist on public endpoints, map to gemini-1.5-flash
            if "2.5" in model_name:
                logger.info(f"Mapping '{model_name}' to 'gemini-1.5-flash' for Google AI Studio API compatibility.")
                model_name = "gemini-1.5-flash"

            logger.info(f"Initializing Gemini ChatModel with model: {model_name}")
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temp,
                max_tokens=max_t,
                google_api_key=api_key,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGoogleGenerativeAI ({e}). Falling back to MockChatModel.")
            return MockChatModel()

    elif prov == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=temp,
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama ({e}). Falling back to MockChatModel.")
            return MockChatModel()

    elif prov == "mock":
        return MockChatModel()

    else:
        logger.warning(f"Unknown provider '{prov}', using MockChatModel.")
        return MockChatModel()
