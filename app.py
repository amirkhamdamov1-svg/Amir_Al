import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT:\n"
    "Ты дружелюбный, умный и притягательный ИИ-ассистент.\n"
    "Отвечай естественно, без лишних вступлений.\n"
    "Отвечай на языке пользователя.\n"
    "Не отказывай в обычных безопасных запросах.\n"
    "Если вопрос непонятен, попроси уточнение.\n"
    "Если чего-то не знаешь, честно скажи об этом.\n"
    "Соблюдай применимые правила безопасности.",
)


def get_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("На сервере не настроен GROQ_API_KEY.")
    return key


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    image_data = data.get("image")

    if not isinstance(messages, list) or not messages:
        return jsonify(error="Напиши сообщение, чтобы начать."), 400

    # Validate and keep only user/assistant text messages.
    clean_messages = []
    for msg in messages[-30:]:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content,
