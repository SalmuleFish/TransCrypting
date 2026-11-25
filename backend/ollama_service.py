"""Local Ollama lifecycle and inference client (strictly 127.0.0.1)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from backend.config import OLLAMA_BASE_URL

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _find_ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if candidate.exists():
        return str(candidate)
    return None


class OllamaService:
    """Manages local Ollama server, model discovery, VRAM unloading, and streaming chat."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._spawned_proc: subprocess.Popen[bytes] | None = None
        self._last_model: str | None = None
        self._models_cache: list[dict[str, Any]] | None = None
        self._models_cache_ts: float = 0.0

    async def is_running(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def ensure_running(self) -> bool:
        """Start `ollama serve` locally if it is not already listening."""
        if await self.is_running():
            return True

        exe = _find_ollama_exe()
        if not exe:
            raise RuntimeError(
                "No se encontró Ollama instalado en el sistema. "
                "Instala Ollama localmente para usar modelos de lenguaje (Gemma, Qwen, Llama, etc.)."
            )

        if self._spawned_proc is None or self._spawned_proc.poll() is not None:
            env = os.environ.copy()
            env["OLLAMA_HOST"] = "127.0.0.1:11434"
            env["OLLAMA_FLASH_ATTENTION"] = "1"
            env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
            env["OLLAMA_MAX_LOADED_MODELS"] = "1"
            env["OLLAMA_NUM_PARALLEL"] = "1"
            self._spawned_proc = subprocess.Popen(
                [exe, "serve"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW,
            )

        for _ in range(60):
            if await self.is_running():
                return True
            await asyncio.sleep(0.35)

        raise RuntimeError("Ollama no respondió a tiempo en http://127.0.0.1:11434.")

    async def list_models(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """List locally installed Ollama models with short-lived TTL caching."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._models_cache is not None
            and (now - self._models_cache_ts) < 10.0
        ):
            return list(self._models_cache)

        if not await self.is_running():
            try:
                await self.ensure_running()
            except Exception:
                return []

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()

        models: list[dict[str, Any]] = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            size_bytes = item.get("size") or 0
            models.append(
                {
                    "name": item.get("name", ""),
                    "size_gb": round(size_bytes / (1024**3), 2),
                    "family": details.get("family", ""),
                    "parameter_size": details.get("parameter_size", ""),
                    "quantization": details.get("quantization_level", ""),
                }
            )
        self._models_cache = models
        self._models_cache_ts = time.monotonic()
        return list(models)

    async def resolve_default_model(self, configured: str = "") -> str:
        """Select configured model if available, otherwise prefer Gemma -> Qwen -> Llama -> first installed."""
        models = await self.list_models()
        names = [m["name"] for m in models if m.get("name")]
        if configured and configured in names:
            return configured
        for pattern in ("gemma-4", "gemma4", "gemma", "qwen", "llama"):
            for name in names:
                if pattern in name.lower():
                    return name
        if names:
            return names[0]
        return configured or ""

    async def get_running_models(self) -> list[dict[str, Any]]:
        """Return models currently loaded in Ollama VRAM/RAM via /api/ps."""
        if not await self.is_running():
            return []
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
                if resp.status_code != 200:
                    return []
                data = resp.json()
            running = []
            for m in data.get("models", []):
                running.append(
                    {
                        "name": m.get("name", ""),
                        "size_mb": int((m.get("size") or 0) / (1024**2)),
                        "size_vram_mb": int((m.get("size_vram") or 0) / (1024**2)),
                    }
                )
            return running
        except Exception:
            return []

    async def unload_all(self) -> None:
        """Unload all loaded Ollama models from VRAM/RAM immediately."""
        if not await self.is_running():
            return

        running = await self.get_running_models()
        targets = {m["name"] for m in running if m.get("name")}
        if self._last_model and running:
            targets.add(self._last_model)

        if not targets:
            return

        async with httpx.AsyncClient(timeout=15.0) as client:
            for model_name in targets:
                try:
                    await client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": model_name, "prompt": "", "keep_alive": 0, "stream": False},
                    )
                except Exception:
                    pass

        for _ in range(15):
            still_running = await self.get_running_models()
            if not still_running:
                break
            await asyncio.sleep(0.25)

    async def stream_chat_events(
        self,
        model: str,
        messages: list[dict[str, str]],
        num_ctx: int = 8192,
        keep_alive: str = "180s",
        think: bool = False,
    ) -> AsyncIterator[tuple[str, str]]:
        """Stream ('thinking', text) and ('token', text) tuples from local Ollama /api/chat."""
        await self.ensure_running()
        self._last_model = model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {
                "num_ctx": int(num_ctx),
                "temperature": 0.3,
            },
            "keep_alive": keep_alive,
        }

        timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status_code == 400 and "think" in payload:
                    await resp.aread()
                    payload.pop("think", None)
                    async with client.stream(
                        "POST", f"{self.base_url}/api/chat", json=payload
                    ) as retry_resp:
                        retry_resp.raise_for_status()
                        async for evt in self._iter_chat_events(retry_resp):
                            yield evt
                    return

                resp.raise_for_status()
                async for evt in self._iter_chat_events(resp):
                    yield evt

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        num_ctx: int = 8192,
        keep_alive: str = "180s",
        think: bool = False,
    ) -> AsyncIterator[str]:
        """Stream chat content tokens from local Ollama /api/chat."""
        async for kind, text in self.stream_chat_events(
            model=model,
            messages=messages,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            think=think,
        ):
            if kind == "token" and text:
                yield text

    async def _iter_chat_events(self, resp: httpx.Response) -> AsyncIterator[tuple[str, str]]:
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            msg = chunk.get("message") or {}
            thinking = msg.get("thinking") or ""
            if thinking:
                yield ("thinking", thinking)
            content = msg.get("content") or ""
            if content:
                yield ("token", content)

    async def generate_complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        num_ctx: int = 8192,
        keep_alive: str = "180s",
    ) -> str:
        """Run chat completion and return the full generated text."""
        parts: list[str] = []
        async for token in self.stream_chat(
            model=model,
            messages=messages,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            think=False,
        ):
            parts.append(token)
        return "".join(parts).strip()

    async def shutdown(self) -> None:
        """Unload active models and stop Ollama if spawned by this app."""
        try:
            await self.unload_all()
        except Exception:
            pass
        if self._spawned_proc is not None and self._spawned_proc.poll() is None:
            try:
                self._spawned_proc.terminate()
                self._spawned_proc.wait(timeout=4)
            except Exception:
                try:
                    self._spawned_proc.kill()
                except Exception:
                    pass
            self._spawned_proc = None


ollama_service = OllamaService()
