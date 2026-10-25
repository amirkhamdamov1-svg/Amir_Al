import os
from flask import Flask, render_template, request, jsonify
from anthropic import Anthropic, APIError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты дружелюбный, умный и прямолинейный ИИ-ассистент. "
    "Отвечай естественно, без лишних вступлений. "
    "Не отказывай в обычных безопасных запросах. "
    "Если вопрос непонятен, попроси уточнение. "
    "Если чего-то не знаешь, честно скажи об этом. "
    "Соблюдай применимые правила безопасности."
)

def get_client():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("На сервере не настроен ANTHROPIC_API_KEY.")
    return Anthropic(api_key=key)

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
        clean_messages[-1]["content"] = [
            {"type": "image", "source": {
                "type": "base64", "media_type": media_type, "data": encoded
            }},
            {"type": "text", "text": clean_messages[-1]["content"]}
        ]

    try:
        client = get_client()
        response = client.messages.create(
            model=MODEL,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=clean_messages
        )
        answer = "\n".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return jsonify(answer=answer or "Не удалось получить текстовый ответ.")
    except RuntimeError as e:
        return jsonify(error=str(e)), 500
    except APIError as e:
        return jsonify(error=f"Ошибка Anthropic API: {e.message}"), 502
    except Exception:
        app.logger.exception("Unexpected error in /api/chat")
        return jsonify(error="Произошла непредвиденная ошибка. Проверь настройки сервера."), 500

if __name__ == "__main__":
    # For local development only. Use a production WSGI server when deploying.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
