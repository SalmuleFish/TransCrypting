"""DeepSeek Flash credentials and chat streaming."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from backend.config import DATA_DIR


DEEPSEEK_MODEL = "deepseek-flash"
_KEY_PATH = DATA_DIR / ".deepseek-api-key"
_CHAT_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekError(RuntimeError):
    pass


def get_api_key() -> str:
    try:
        return _KEY_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def save_api_key(key: str) -> None:
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=_KEY_PATH.parent, delete=False) as file:
        temp_path = Path(file.name)
        try:
            file.write(key)
            file.close()
            temp_path.replace(_KEY_PATH)
        finally:
            temp_path.unlink(missing_ok=True)


async def stream_events(
    client: httpx.AsyncClient,
    key: str,
    messages: list[dict[str, Any]],
    think: bool,
    max_tokens: int | None,
) -> AsyncIterator[tuple[str, str]]:
    payload: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": True,
        "thinking": {"type": "enabled" if think else "disabled"},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    errors = {
        401: "DeepSeek rechazó la clave API. Revísala en Configuración → Texto.",
        402: "Tu cuenta de DeepSeek no tiene saldo suficiente.",
        429: "DeepSeek alcanzó su límite de solicitudes. Intenta de nuevo en un momento.",
    }
    try:
        async with client.stream(
            "POST", _CHAT_URL, headers={"Authorization": f"Bearer {key}"}, json=payload,
        ) as response:
            if response.is_error:
                raise DeepSeekError(errors.get(response.status_code, f"DeepSeek respondió con error HTTP {response.status_code}."))
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise DeepSeekError("DeepSeek interrumpió la respuesta.")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield "thinking", delta["reasoning_content"]
                    if delta.get("content"):
                        yield "token", delta["content"]
            raise DeepSeekError("La conexión con DeepSeek se cerró antes de terminar la respuesta.")
    except httpx.HTTPError as exc:
        raise DeepSeekError("No se pudo completar la conexión con DeepSeek.") from exc
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise DeepSeekError("DeepSeek devolvió una respuesta inesperada.") from exc
