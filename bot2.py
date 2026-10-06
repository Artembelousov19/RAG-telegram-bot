import sys
import types
import os
import json
import time
import html
import urllib.request
import urllib.parse
from dotenv import load_dotenv
import chromadb
from chromadb.api.types import EmbeddingFunction

# 1. Загружаем переменные окружения из локального файла .env
load_dotenv()

# 2. Безопасное получение токена и конфигурации из окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db")

# Проверка безопасности: останавливаем выполнение, если токен не задан
if not BOT_TOKEN:
    raise ValueError(
        "ОШИБКА: Переменная BOT_TOKEN не найдена!\n"
        "Создайте файл .env в корневой папке проекта и добавьте строку:\n"
        "BOT_TOKEN=your_telegram_bot_token_here"
    )

# Сохраняем обратную совместимость с pydantic_settings при необходимости
try:
    import pydantic_settings
except ImportError:
    pydantic_settings = types.ModuleType("pydantic_settings")
    pydantic_settings.BaseSettings = object
    sys.modules["pydantic_settings"] = pydantic_settings


# Dummy Embedding Function для подключения к существующей ChromaDB
class DummyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: list) -> list:
        return [[0.0] * 384 for _ in input]


# Сброс прокси-серверов для локальных сетевых запросов к Telegram API
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

null_proxy_handler = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(null_proxy_handler)
urllib.request.install_opener(opener)

# Инициализация векторной базы данных ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
collection = chroma_client.get_or_create_collection(
    name="agritech_docs",
    embedding_function=DummyEmbeddingFunction()
)


def query_ollama(prompt: str) -> str:
    """Отправка запроса к локальной Ollama LLM."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "Ошибка: пустой ответ от LLM.")
    except Exception as e:
        return f"Ошибка при обращении к Ollama: {e}"


def get_telegram_updates(offset: int = None) -> dict:
    """Получение новых сообщений через Telegram Long Polling."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?timeout=30"
    if offset:
        url += f"&offset={offset}"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=35) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Ошибка соединения с Telegram: {e}")
        return {"ok": False, "result": []}


def send_telegram_message(chat_id: int, text: str):
    """Отправка ответа пользователю с безопасной HTML-разметкой."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Ошибка при отправке сообщения в чат {chat_id}: {e}")


def handle_rag_query(user_query: str) -> str:
    """Основной RAG-пайплайн: поиск релевантных чанков + генерация ответа."""
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )
    
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if not docs:
        context_str = "Контекст не найден в базе знаний."
        refs_str = "Источники не найдены."
    else:
        context_str = "\n\n".join(docs)
        sources = [meta.get("source", "doc_unknown") for meta in metadatas if meta]
        unique_sources = list(dict.fromkeys(sources))
        refs_str = "\n".join([f"• <code>{html.escape(src)}</code>" for src in unique_sources])

    prompt = (
        f"You are a technical support assistant for agricultural robotics.\n"
        f"Answer the user query based ONLY on the context below.\n\n"
        f"Context:\n{context_str}\n\n"
        f"User Query: {user_query}\n\n"
        f"Answer:"
    )
    
    raw_answer = query_ollama(prompt)
    clean_answer = html.escape(raw_answer)
    
    formatted_response = (
        f"🤖 <b>Agritech Support Assistant</b>\n\n"
        f"{clean_answer}\n\n"
        f"📚 <b>References:</b>\n{refs_str}"
    )
    return formatted_response


def main():
    print("=== Telegram RAG Bot успешно запущен ===")
    offset = None
    
    while True:
        updates = get_telegram_updates(offset)
        if updates.get("ok"):
            for update in updates.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message", {})
                chat_id = message.get("chat", {}).get("id")
                text = message.get("text", "")
                
                if not chat_id or not text:
                    continue
                
                if text == "/start":
                    welcome_msg = (
                        "👋 <b>Welcome to Agritech RAG Assistant!</b>\n\n"
                        "Ask any question about harvesting robots, sensor calibration, "
                        "or technical fault codes."
                    )
                    send_telegram_message(chat_id, welcome_msg)
                    continue
                
                print(f"Получен запрос от [{chat_id}]: {text}")
                response = handle_rag_query(text)
                send_telegram_message(chat_id, response)
                
        time.sleep(1)


if __name__ == "__main__":
    main()