import logging
import tempfile
from pathlib import Path

import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest

from rag import RAGEngine
from config import BOT_TOKEN, OPENAI_API_KEY, SYSTEM_PROMPT, RAG_TEMPLATE, LLM_MODEL, OPENAI_BASE_URL

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

rag = RAGEngine()
openai_client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 RAG-бот\n\n"
        "Я отвечаю на вопросы на основе загруженных документов.\n\n"
        "📄 Отправь мне текстовый файл (.txt, .md) — я его запомню\n"
        "❓ Задай вопрос — я найду ответ в документах\n\n"
        "Команды:\n"
        "/start — это сообщение\n"
        "/clear — очистить все документы\n"
        "/stats — статистика"
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = rag.count_documents()
    await update.message.reply_text(f"📊 Всего чанков в базе: {count}")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rag.clear()
    await update.message.reply_text("🗑 База знаний очищена")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.file_name and not doc.file_name.endswith((".txt", ".md", ".csv", ".json", ".yaml", ".yml")):
        await update.message.reply_text("Поддерживаются только .txt, .md, .csv, .json, .yaml")
        return

    msg = await update.message.reply_text("⏳ Обрабатываю документ...")

    file = await doc.get_file()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        tmp_path = f.name

    try:
        await file.download_to_drive(tmp_path)
        raw = Path(tmp_path).read_bytes()
        encodings = ["utf-8", "windows-1251", "cp866", "koi8-r"]
        text = None
        for enc in encodings:
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")

        if len(text) < 20:
            await msg.edit_text("Файл слишком короткий (< 20 символов)")
            return

        doc_id = rag.add_document(text, {"filename": doc.file_name})
        chunks = rag.count_documents()
        await msg.edit_text(
            f"✅ Документ '{doc.file_name}' загружен\n"
            f"🆔 ID: {doc_id}\n"
            f"🧩 Всего чанков: {chunks}"
        )
    except Exception as e:
        logger.exception("File processing error")
        await msg.edit_text(f"❌ Ошибка: {e}")
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    msg = await update.message.reply_text("🔍 Ищу ответ...")

    results = rag.search(query, k=5)
    if not results:
        await msg.edit_text("База знаний пуста. Сначала загрузи документы.")
        return

    context_text = "\n\n".join(
        f"[{r['metadata'].get('filename', 'unknown')}] {r['text']}"
        for r in results
    )

    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": RAG_TEMPLATE.format(
                        context=context_text, question=query
                    )},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.exception("LLM error")
            answer = None
    else:
        answer = None

    if not answer:
        top = results[0]
        answer = (
            f"📄 {top['metadata'].get('filename', 'документ')}:\n\n"
            f"{top['text'][:1500]}"
        )

    sources = "\n".join(
        f"📎 {r['metadata'].get('filename', 'документ')} ({r['score']:.2f})"
        for r in results
    )

    await msg.edit_text(f"{answer}\n\n{sources}")


def _build_app():
    builder = Application.builder().token(BOT_TOKEN)
    proxy = os.getenv("TG_PROXY", "")
    if proxy:
        request = HTTPXRequest(proxy_url=proxy)
        builder = builder.request(request)
    return builder.build()


def main():
    if not BOT_TOKEN:
        logger.error("TG_BOT_TOKEN не задан! Укажи токен в .env или переменной окружения")
        return

    app = _build_app()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
