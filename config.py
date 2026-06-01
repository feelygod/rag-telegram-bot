import os
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()


def get_bot_token() -> str:
    return os.getenv("TG_BOT_TOKEN", "")


def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


OPENAI_API_KEY = get_openai_key()
OPENAI_BASE_URL = get_openai_base_url()
LLM_MODEL = get_llm_model()

SYSTEM_PROMPT = (
    "Ты — AI-ассистент, отвечающий на вопросы на основе предоставленных документов. "
    "Отвечай кратко, по делу, только на основе контекста. "
    "Если в контексте нет ответа — скажи, что не знаешь. "
    "Отвечай на том же языке, на котором задан вопрос."
)

RAG_TEMPLATE = "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
