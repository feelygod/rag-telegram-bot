import os

BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "Ты — AI-ассистент, отвечающий на вопросы на основе предоставленных документов. "
    "Отвечай кратко, по делу, только на основе контекста. "
    "Если в контексте нет ответа — скажи, что не знаешь. "
    "Отвечай на том же языке, на котором задан вопрос."
)

RAG_TEMPLATE = "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
