from __future__ import annotations

from typing import Any

from .config import (
    API_KEY,
    BASE_URL,
    LLM_BACKEND,
    LLM_CTX,
    LLM_GPU_LAYERS,
    LLM_MODEL_PATH,
    LLM_THREADS,
)
from .debug import debug_context, debug_event, info_event
from .llama_server import ensure_llama_server_running


class LLMClient:
    """Tiny wrapper that supports:

    - LLM_BACKEND=openai_compatible: OpenAI-compatible endpoint (llama.cpp server, vLLM, etc.)
    - LLM_BACKEND=llama_cpp: in-process GGUF model via llama-cpp-python

    The rest of the agent expects a .chat(...) method that returns *text* in response.
    """

    def __init__(self) -> None:
        self.backend = LLM_BACKEND
        self.model: str | None = None

        self._openai_client = None
        self._llama = None

        if self.backend == "llama_cpp":
            self._init_llama_cpp()
        else:
            self._init_openai_compatible()

    def _init_openai_compatible(self) -> None:
        from openai import OpenAI  # local import (optional dependency)

        self._openai_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def _init_llama_cpp(self) -> None:
        if not LLM_MODEL_PATH:
            raise RuntimeError(
                "LLM_BACKEND=llama_cpp requires LLM_MODEL_PATH env var pointing to a .gguf model"
            )
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it or set LLM_BACKEND=openai_compatible"
            ) from exc

        kwargs: dict[str, Any] = {
            "model_path": LLM_MODEL_PATH,
            "n_ctx": LLM_CTX,
        }
        if LLM_THREADS > 0:
            kwargs["n_threads"] = LLM_THREADS
        if LLM_GPU_LAYERS > 0:
            kwargs["n_gpu_layers"] = LLM_GPU_LAYERS

        self._llama = Llama(**kwargs)

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        # Keep a simple "role+content" schema.
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content") or "")
            if role not in {"system", "user", "assistant"}:
                role = "user"
                content = f"[{str(message.get('role', '')).upper()}]\n{content}".strip()
            normalized.append({"role": role, "content": content})
        return normalized

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,  # kept for compatibility, may be unused
        model_name: str,
        tool_choice: str = "auto",
    ) -> Any:
        normalized = self._normalize_messages(messages)
        debug_event("LLM_REQ", f"backend={self.backend} model={model_name or ''} tool_choice={tool_choice}")
        debug_context("LLM_REQ", normalized, limit=1200)
        info_event("LLM_REQ_FULL", str(normalized))

        if self.backend == "llama_cpp":
            # In-process: we ask for plain text (the agent parses code blocks itself).
            assert self._llama is not None
            out = self._llama.create_chat_completion(
                messages=normalized,
                temperature=0.2,
            )
            content = out["choices"][0]["message"]["content"] or ""
            debug_context("LLM_RES", content, limit=1200)
            info_event("LLM_RES_FULL", content)
            return out

        # OpenAI-compatible endpoint
        assert self._openai_client is not None
        model_name = self._resolve_model_name(model_name)
        self.model = model_name
        ensure_llama_server_running()
        response = self._openai_client.chat.completions.create(
            model=model_name,
            messages=normalized,
        )
        content = response.choices[0].message.content or ""
        debug_context("LLM_RES", content, limit=1200)
        info_event("LLM_RES_FULL", content)
        return response

    def _resolve_model_name(self, model_name: str) -> str:
        if model_name:
            return model_name
        if self.model:
            return self.model
        try:
            assert self._openai_client is not None
            models = self._openai_client.models.list()
            if models.data:
                return models.data[0].id
        except Exception:
            pass
        return "default"