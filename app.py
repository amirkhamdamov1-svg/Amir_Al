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
        if isinstance(content, str) and content.strip():
            clean_messages.append({"role": msg["role"], "content": content[:12000]})

    if not clean_messages or clean_messages[-1]["role"] != "user":
        return jsonify(error="Последнее сообщение должно быть сообщением пользователя."), 400

    model = GROQ_VISION_MODEL

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
            {
                "type": "text",
                "text": last_text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{encoded}"
                },
            },
        ]
        model = GROQ_VISION_MODEL

    try:
        api_key = get_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + clean_messages

        payload = {
            "model": model,
            "max_tokens": 1200,
            "messages": payload_messages,
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 429:
            return jsonify(error="Превышен лимит запросов к Groq API (Rate Limit). Попробуйте позже."), 429

        if response.status_code != 200:
            return jsonify(error=f"Ошибка Groq API: {response.text}"), 502

        res_json = response.json()
        answer = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        return jsonify(answer=answer or "Не удалось получить текстовый ответ.")
    except RuntimeError as e:
        return jsonify(error=str(e)), 500
    except Exception:
        app.logger.exception("Unexpected error in /api/chat")
        return jsonify(error="Произошла непредвиденная ошибка. Проверь настройки сервера."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
