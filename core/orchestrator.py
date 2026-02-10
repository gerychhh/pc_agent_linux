from __future__ import annotations

import re
import time
import os
from typing import Any, Callable

from .command_engine import CommandResult, load_commands, match_command, run_command
from .config import FAST_MODEL, TIMEOUT_SEC
from .context_builder import ctx_action, ctx_action_repair, ctx_reporter
from .debug import debug_event, truncate_text
from .executor import run_python, run_shell
from .interaction_memory import find_similar_actions, get_action, set_action
from .llm_client import LLMClient
from .llm_parser import parse_action_from_text
from .state import load_state
from .validator import validate_bash, validate_python

BLOCKED_MESSAGE = "Команда заблокирована по безопасности (опасная операция)."


def _truthy(v: str) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    starters = (
        "кто",
        "что",
        "почему",
        "зачем",
        "как",
        "какой",
        "какая",
        "какие",
        "где",
        "когда",
        "сколько",
        "чем",
        "объясни",
        "расскажи",
        "tell me",
        "what",
        "who",
        "why",
        "how",
        "where",
        "when",
    )
    return t.startswith(starters)


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("[TOOL_RESULT]", "").replace("[/TOOL_RESULT]", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _format_script_action(lang: str, script: str) -> str:
    s = (script or "").strip()
    if "\n" in s:
        s = s.splitlines()[0].strip() + " ..."
    return f"SCRIPT[{lang}]: {s}"


def _format_simple_response(result: dict[str, Any]) -> str:
    if not result:
        return "NEED_CONFIRM"
    ok = bool(result.get("ok"))
    out = (result.get("stdout") or "").strip()
    err = (result.get("stderr") or "").strip()
    if ok and out:
        return out
    if ok and not out and err:
        return err
    if ok and not out and not err:
        return "OK"
    if (not ok) and err:
        return f"ERROR: {err}"
    if (not ok) and out:
        return f"ERROR: {out}"
    return "NEED_CONFIRM"


def _format_action_output(action_desc: str, output: str) -> str:
    if not action_desc:
        return output
    if not output:
        return action_desc
    return f"{action_desc}\n{output}"


def _call_with_timeout(func: Callable[..., Any], script: str, timeout_sec: int) -> Any:
    """
    Compatibility wrapper for run_shell/run_python, because project versions differ.

    Tries:
      - func(script, timeout_sec=...)
      - func(script, timeout=...)
      - func(script, timeout_sec)   (positional)
      - func(script)
    """
    for kwargs in ({"timeout_sec": timeout_sec}, {"timeout": timeout_sec}):
        try:
            return func(script, **kwargs)
        except TypeError:
            pass
    try:
        return func(script, timeout_sec)  # positional
    except TypeError:
        return func(script)


def _validate_and_exec(lang: str, script: str) -> tuple[dict[str, Any], str]:
    lang = (lang or "").strip().lower()
    if lang in ("bash", "sh", "shell", "zsh"):
        lang = "bash"
    elif lang in ("py", "python3"):
        lang = "python"

    if lang == "bash":
        blocked = validate_bash(script)
        if blocked:
            return {"ok": False, "blocked": True, "stdout": "", "stderr": blocked}, "bash"
        res = _call_with_timeout(run_shell, script, TIMEOUT_SEC)
        return res, "bash"

    if lang == "python":
        blocked = validate_python(script)
        if blocked:
            return {"ok": False, "blocked": True, "stdout": "", "stderr": blocked}, "python"
        res = _call_with_timeout(run_python, script, TIMEOUT_SEC)
        return res, "python"

    return {"ok": False, "blocked": True, "stdout": "", "stderr": "Unknown language"}, lang or "unknown"


class Orchestrator:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.commands = load_commands()

    def reset(self) -> None:
        return None

    def run(self, user_text: str, stateless: bool = False, force_llm: bool = False) -> str:
        state = load_state()
        debug_event("USER_IN", user_text)

        # Questions -> text only (no execution)
        if _looks_like_question(user_text):
            out = self._run_llm_text(user_text, state)
            if out is None:
                return "Не удалось получить ответ от модели."
            text, llm_ms = out
            if _truthy(os.getenv("PC_AGENT_SHOW_LLM_TEXT", "1")):
                return f"{text}\nLLM time: {llm_ms} ms"
            return f"LLM time: {llm_ms} ms"

        # 0) Learned action fast path
        learned = get_action(user_text)
        if not learned:
            similar = find_similar_actions(user_text, limit=1)
            if similar and similar[0]["score"] >= 0.94:
                learned = similar[0]["action"]

        if learned and not force_llm:
            out = self._run_learned_action(learned)
            if out:
                return out

        # 1) Command library
        if not force_llm:
            match, _ = match_command(user_text, self.commands)
            if match:
                result: CommandResult = run_command(match.command, match.params)
                if result.ok:
                    set_action(user_text, {"type": "cmd", "id": match.command.get("id"), "params": match.params or {}})
                return result.to_text()

        # 2) LLM fallback -> script
        llm_action = self._run_llm_script(user_text, state)
        if llm_action is None:
            return "Не удалось подобрать команду."

        exec_result, action_desc, learned_action, llm_ms, llm_text = llm_action

        if exec_result is None:
            msg = "Не удалось подобрать команду."
            if _truthy(os.getenv("PC_AGENT_SHOW_LLM_TEXT", "1")) and llm_text:
                msg += "\nLLM: " + llm_text
            msg += f"\nLLM time: {llm_ms} ms"
            return msg

        if exec_result.get("blocked"):
            msg = BLOCKED_MESSAGE
            if action_desc:
                msg += "\n" + action_desc
            msg += "\n" + _format_simple_response(exec_result)
            if _truthy(os.getenv("PC_AGENT_SHOW_LLM_TEXT", "1")) and llm_text:
                msg += "\nLLM: " + llm_text
            msg += f"\nLLM time: {llm_ms} ms"
            return msg

        summarized = {
            "ok": bool(exec_result.get("ok")),
            "stdout": (exec_result.get("stdout") or "").strip(),
            "stderr": (exec_result.get("stderr") or "").strip(),
        }
        response = _format_action_output(action_desc, _format_simple_response(summarized))

        if _truthy(os.getenv("PC_AGENT_SHOW_LLM_TEXT", "1")) and llm_text:
            response += "\nLLM: " + llm_text

        response += f"\nLLM time: {llm_ms} ms"

        if summarized.get("ok") and learned_action:
            set_action(user_text, learned_action)

        return response

    def _run_learned_action(self, learned_action: dict[str, Any]) -> str | None:
        try:
            if learned_action.get("type") == "cmd":
                cmd_id = learned_action.get("id")
                params = learned_action.get("params") or {}
                match = next((c for c in self.commands if (c.get("id") == cmd_id)), None)
                if not match:
                    return None
                result: CommandResult = run_command(match, params)
                return result.to_text() if result.ok else None

            if learned_action.get("type") == "script":
                lang = learned_action.get("lang", "bash")
                script = learned_action.get("script", "")
                if not script:
                    return None
                exec_result, normalized = _validate_and_exec(lang, script)
                if exec_result.get("blocked"):
                    return BLOCKED_MESSAGE
                action_desc = _format_script_action(normalized, script)
                summarized = {
                    "ok": bool(exec_result.get("ok")),
                    "stdout": (exec_result.get("stdout") or "").strip(),
                    "stderr": (exec_result.get("stderr") or "").strip(),
                }
                return _format_action_output(action_desc, _format_simple_response(summarized))
        except Exception:
            return None
        return None

    def _run_llm_text(self, user_text: str, state: dict[str, Any]) -> tuple[str, int] | None:
        payload = ctx_reporter(state, user_text)
        debug_event("LLM_REQ", f"report model={FAST_MODEL} payload={truncate_text(payload, 400)}")

        t0 = time.monotonic()
        try:
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=FAST_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            debug_event("LLM_ERR", str(exc))
            return None

        llm_ms = int((time.monotonic() - t0) * 1000)
        return sanitize_assistant_text(content), llm_ms

    def _run_llm_script(
        self, user_text: str, state: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None, int, str] | None:
        payload = ctx_action(state, user_text)
        debug_event("LLM_REQ", f"action model={FAST_MODEL} payload={truncate_text(payload, 400)}")

        t0 = time.monotonic()
        try:
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=FAST_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            debug_event("LLM_ERR", str(exc))
            return None

        llm_text = sanitize_assistant_text(content)
        llm_ms = int((time.monotonic() - t0) * 1000)

        parsed = parse_action_from_text(content)
        if not parsed:
            repaired = self._repair_llm_script(user_text, state, llm_text)
            if repaired is not None:
                exec_result, action_desc, learned_action, repair_ms, repair_text = repaired
                return exec_result, action_desc, learned_action, llm_ms + repair_ms, repair_text or llm_text
            return None, "", None, llm_ms, llm_text

        exec_result, normalized_lang = _validate_and_exec(parsed.language, parsed.script)
        action_desc = _format_script_action(normalized_lang, parsed.script)

        learned_action = None
        if exec_result is not None and not exec_result.get("blocked"):
            learned_action = {"type": "script", "lang": normalized_lang, "script": parsed.script}

        return exec_result, action_desc, learned_action, llm_ms, llm_text

    def _repair_llm_script(
        self, user_text: str, state: dict[str, Any], llm_text: str
    ) -> tuple[dict[str, Any], str, dict[str, Any] | None, int, str] | None:
        if not llm_text:
            return None
        payload = ctx_action_repair(state, user_text, llm_text)
        debug_event("LLM_REQ", f"repair model={FAST_MODEL} payload={truncate_text(payload, 400)}")

        t0 = time.monotonic()
        try:
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=FAST_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            debug_event("LLM_ERR", str(exc))
            return None

        repair_text = sanitize_assistant_text(content)
        repair_ms = int((time.monotonic() - t0) * 1000)

        parsed = parse_action_from_text(content)
        if not parsed:
            return None

        exec_result, normalized_lang = _validate_and_exec(parsed.language, parsed.script)
        action_desc = _format_script_action(normalized_lang, parsed.script)

        learned_action = None
        if exec_result is not None and not exec_result.get("blocked"):
            learned_action = {"type": "script", "lang": normalized_lang, "script": parsed.script}

        return exec_result, action_desc, learned_action, repair_ms, repair_text
