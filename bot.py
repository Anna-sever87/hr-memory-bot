import asyncio
import hashlib
import logging
import os
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Deque, Dict, List, TypedDict

import chromadb
import docx  # библиотека python-docx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Читаем настройки из файла .env (значения из .env главнее системных переменных)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# -----------------------------
# Константы и настройки
# -----------------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

PERSIST_DIR = "./memory"    # папка векторной базы Chroma (долгая память)
UPLOADS_DIR = "./uploads"   # папка для загруженных файлов
COLLECTION_NAME = "long_memory"

CHUNK_SIZE = 500            # размер чанка в символах
CHUNK_OVERLAP = 50          # перекрытие между чанками
MAX_CHUNKS = 2000           # ограничение на размер одного документа
EMBED_BATCH = 100           # сколько чанков отправляем на эмбеддинг за раз
TOP_K = 5                   # сколько фрагментов достаём из базы
RELEVANCE_MAX_DISTANCE = 0.8  # чем меньше расстояние, тем фрагмент ближе по смыслу

SHORT_MEMORY_SIZE = 10      # сколько последних сообщений помним (короткая память)
MAX_FILE_SIZE = 20 * 1024 * 1024  # лимит Telegram на скачивание файлов ботом
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".docx")

SYSTEM_PROMPT = (
    "Ты — дружелюбный HR-ассистент компании. Отвечай на русском языке, кратко и по делу, на «вы».\n"
    "Если в сообщении есть блок «Контекст из документов», отвечай прежде всего на его основе. "
    "Числа, сроки и условия бери только из этого контекста.\n"
    "Если вопрос касается документов, а ответа в контексте нет, честно скажи, что не нашёл этого "
    "в загруженных документах. Не выдумывай факты.\n"
    "Для уточняющих вопросов используй историю диалога.\n"
    "Если вопрос общий и не связан с документами, отвечай кратко и по здравому смыслу."
)


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


# Короткая память: user_id -> кольцевая очередь последних сообщений (в оперативной памяти)
short_memory: Dict[int, Deque[ChatMessage]] = defaultdict(
    lambda: deque(maxlen=SHORT_MEMORY_SIZE)
)

router = Router()


# -----------------------------
# Служебные функции
# -----------------------------
def setup_logger() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    # Убираем лишние технические сообщения библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Не найдена переменная {var_name}. Проверь файл .env")
    return value


def ensure_dirs() -> None:
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Создаёт клиент OpenAI один раз. Ключ и адрес берутся из .env."""
    return OpenAI(
        api_key=require_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


@lru_cache(maxsize=1)
def get_collection():
    """Возвращает (или создаёт) коллекцию Chroma с постоянным хранилищем на диске."""
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


async def send_long(message: Message, text: str) -> None:
    """Отправляет длинный ответ частями (лимит Telegram — 4096 символов)."""
    text = text.strip() or "Пустой ответ."
    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000])


# -----------------------------
# Долгая память: документы -> чанки -> эмбеддинги -> Chroma
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Разбивает текст на пересекающиеся чанки примерно по chunk_size символов."""
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def load_document(file_path: str) -> List[str]:
    """Загружает документ (PDF, TXT, DOCX), извлекает текст и возвращает список чанков."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
    elif suffix == ".docx":
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(parts)
    else:
        raise RuntimeError("Поддерживаются только файлы PDF, TXT и DOCX.")

    return chunk_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Получает эмбеддинги (числовое описание смысла) для списка текстов."""
    if not texts:
        return []
    client = get_openai_client()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_chunks(user_id: int, doc_id: str, file_name: str, chunks: List[str]) -> int:
    """Считает эмбеддинги чанков и сохраняет их в Chroma вместе с метаданными."""
    if not chunks:
        return 0
    collection = get_collection()
    vectors = embed_texts(chunks)
    ids = [f"{user_id}:{doc_id}:{i}" for i in range(len(chunks))]  # один файл не дублируется
    metadatas = [
        {"user_id": str(user_id), "doc_id": doc_id, "file_name": file_name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=chunks)
    return len(chunks)


def retrieve_context(user_id: int, question: str, top_k: int = TOP_K) -> List[str]:
    """Ищет в векторной базе фрагменты документов пользователя, близкие по смыслу к вопросу."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    question_vector = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[question_vector],
        n_results=min(top_k, total),
        where={"user_id": str(user_id)},
        include=["documents", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    found: List[str] = []
    for text, distance in zip(documents, distances):
        logger.info("Фрагмент найден, расстояние %.3f", distance)
        if distance <= RELEVANCE_MAX_DISTANCE:
            found.append(text)
    return found


# -----------------------------
# Ответ: системный промпт + короткая память + найденный контекст
# -----------------------------
def answer_question(user_id: int, question: str, context_chunks: List[str]) -> str:
    """Формирует ответ модели, объединяя короткую память (историю) и долгую (контекст)."""
    history = list(short_memory[user_id])

    if context_chunks:
        context_text = "\n\n".join(
            f"Фрагмент {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        user_content = f"Контекст из документов:\n{context_text}\n\nВопрос: {question}"
    else:
        user_content = (
            f"{question}\n\n(Подходящих фрагментов в загруженных документах не найдено.)"
        )

    messages: List[ChatMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]

    completion = get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


def remember(user_id: int, role: str, content: str) -> None:
    """Добавляет реплику в короткую память (старые вытесняются автоматически)."""
    short_memory[user_id].append({"role": role, "content": content})


# -----------------------------
# Обработчики Telegram
# -----------------------------
@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Здравствуйте! Я HR-ассистент с короткой и долгой памятью.\n\n"
        "- Отправьте документ (PDF, DOCX или TXT): я сохраню его в векторной базе (долгая память).\n"
        "- Задавайте вопросы: я отвечу по документам и помню последние 10 реплик диалога "
        "(короткая память).\n\n"
        "Команды:\n"
        "/clear - очистить историю диалога\n"
        "/forget - удалить ваши документы из базы"
    )


@router.message(Command("clear"))
async def on_clear(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    short_memory[user_id].clear()
    await message.answer("История диалога очищена. Документы остались в базе.")


@router.message(Command("forget"))
async def on_forget(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    try:
        await asyncio.to_thread(
            lambda: get_collection().delete(where={"user_id": str(user_id)})
        )
    except Exception:
        logger.exception("Ошибка удаления документов")
        await message.answer("Не удалось удалить документы. Попробуйте позже.")
        return
    await message.answer("Ваши документы удалены из базы.")


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    """Загрузка файла: скачиваем, извлекаем текст, режем на чанки, сохраняем в Chroma."""
    user_id = message.from_user.id if message.from_user else 0
    doc = message.document
    file_name = Path(doc.file_name or "document").name
    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        await message.answer("Поддерживаются только файлы PDF, DOCX и TXT.")
        return
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        await message.answer("Файл слишком большой. Максимум 20 МБ.")
        return

    await message.answer("Читаю документ, подождите немного...")

    ensure_dirs()
    user_dir = Path(UPLOADS_DIR) / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    save_path = user_dir / file_name

    try:
        await bot.download(doc, destination=save_path)
    except Exception:
        logger.exception("Ошибка скачивания файла")
        await message.answer("Не удалось скачать файл. Попробуйте ещё раз.")
        return

    try:
        chunks = await asyncio.to_thread(load_document, str(save_path))
    except Exception:
        logger.exception("Ошибка чтения документа")
        await message.answer("Не удалось прочитать документ. Возможно, файл повреждён.")
        return

    if not chunks:
        await message.answer(
            "Не удалось извлечь текст из документа. Возможно, это скан без текстового слоя."
        )
        return
    if len(chunks) > MAX_CHUNKS:
        await message.answer("Документ слишком большой. Загрузите файл поменьше.")
        return

    doc_id = hashlib.sha1(save_path.read_bytes()).hexdigest()[:16]
    try:
        count = await asyncio.to_thread(embed_chunks, user_id, doc_id, file_name, chunks)
    except Exception:
        logger.exception("Ошибка эмбеддингов или сохранения в Chroma")
        await message.answer("Ошибка при сохранении в векторную базу. Проверьте ключ OpenAI.")
        return

    logger.info("Документ %s сохранён, чанков: %s", file_name, count)
    await message.answer(
        f"Документ «{file_name}» сохранён и проиндексирован. Чанков: {count}. Теперь задайте вопрос."
    )


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, bot: Bot) -> None:
    """Обычный вопрос: ищем контекст в долгой памяти и отвечаем с учётом короткой."""
    user_id = message.from_user.id if message.from_user else 0
    question = (message.text or "").strip()
    if not question:
        return

    await bot.send_chat_action(message.chat.id, "typing")

    # Долгая память: ищем релевантные фрагменты документов
    try:
        context_chunks = await asyncio.to_thread(retrieve_context, user_id, question)
    except Exception:
        logger.exception("Ошибка поиска по базе")
        context_chunks = []  # продолжаем без документов, только с короткой памятью

    # Ответ модели: системный промпт + короткая память + найденный контекст
    try:
        reply = await asyncio.to_thread(answer_question, user_id, question, context_chunks)
    except Exception:
        logger.exception("Ошибка запроса к модели")
        await message.answer("Не удалось получить ответ от модели. Попробуйте позже.")
        return

    # Короткая память: запоминаем сам вопрос и ответ (без вставленного контекста)
    remember(user_id, "user", question)
    remember(user_id, "assistant", reply)

    await send_long(message, reply)


# -----------------------------
# Запуск
# -----------------------------
async def main() -> None:
    setup_logger()
    ensure_dirs()

    # Проверяем настройки сразу, чтобы понятно сообщить об ошибке
    bot_token = require_env("BOT_TOKEN")
    require_env("OPENAI_API_KEY")

    bot = Bot(token=bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Бот запущен. Модель: %s, эмбеддинги: %s. Ожидаю сообщения...", CHAT_MODEL, EMBED_MODEL)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass