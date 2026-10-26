
import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты дружелюбный, умный и прямолинейный ИИ-ассистент. "
    "Отвечай естественно, без лишних вступлений. "
    "Отвечай на языке пользователя. "
    "Не отказывай в обычных безопасных запросах. "
    "Если вопрос непонятен, попроси уточнение. "
    "Если чего-то не знаешь, честно скажи об этом. "
    "Соблюдай применимые правила безопасности.",
)


def get_key():
    key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
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
        if isinstance(content, str) and content.strip():
            clean_messages.append({"role": msg["role"], "content": content[:12000]})

    if not clean_messages or clean_messages[-1]["role"] != "user":
        return jsonify(error="Последнее сообщение должно быть сообщением пользователя."), 400

    model = MODEL

    # Optional image is attached only to the latest user message.
    if image_data:
        if not isinstance(image_data, dict):
            return jsonify(error="Неверный формат изображения."), 400

        media_type = image_data.get("media_type", "")
        encoded = image_data.get("data", "")

        if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
            return jsonify(error="Поддерживаются JPEG, PNG, WEBP и GIF."), 400
        if not isinstance(encoded, str) or len(encoded) > 5_500_000:
            return jsonify(error="Изображение слишком большое."), 400

        last_text = clean_messages[-1]["content"]
        clean_messages[-1]["content"] = [
            {"type": "text", "text": last_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            },
        ]
        model = VISION_MODEL

    try:
        headers = {
            "Authorization": f"Bearer {get_key()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 1200,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + clean_messages,
        }
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)

        if r.status_code == 429:
            return jsonify(error="Слишком много запросов, подожди минуту и попробуй снова."), 429
        if r.status_code != 200:
            app.logger.error("Groq error %s: %s", r.status_code, r.text)
            return jsonify(error=f"Ошибка Groq API ({r.status_code})."), 502

        answer = r.json()["choices"][0]["message"]["content"]
        return jsonify(answer=answer or "Не удалось получить текстовый ответ.")
    except RuntimeError as e:
        return jsonify(error=str(e)), 500
    except requests.Timeout:
        return jsonify(error="Сервер долго не отвечает, попробуй ещё раз."), 504
    except Exception:
        app.logger.exception("Unexpected error in /api/chat")
        return jsonify(error="Произошла непредвиденная ошибка. Проверь настройки сервера."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
