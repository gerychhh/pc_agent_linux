from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass

from .debug import debug_event


@dataclass
class ParsedAction:
    language: str
    script: str


CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+-]+)?\s*\n?(.*?)```", re.DOTALL)

_LANG_MAP = {
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "python": "python",
    "py": "python",
    "python3": "python",
}

_SENTENCE_START_RE = re.compile(
    r"(?i)^(here|note|ok|okay|sure|please|i|we|you|let's|command|run|try|"
    r"сейчас|вот|команда|запусти|запустить|нужно|можно|я|мы|ты|вы|сделаю|делаю)\b"
)
_PROMPT_RE = re.compile(r"^[>$]\s+")
_BARE_START_RE = re.compile(r"^[A-Za-z0-9_./~$]")


def _normalize_lang(lang: str) -> str | None:
    if not lang:
        return "bash"
    return _LANG_MAP.get(lang.strip().lower())


def _parse_bare_script(text: str) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if "```" in cleaned:
        return None
    if "NEED_CONFIRM" in cleaned:
        return None

    lines: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _PROMPT_RE.sub("", line).strip()
        if not line:
            continue
        if line.startswith("#"):
            lines.append(line)
            continue
        first_char = line[0]
        if ord(first_char) > 127:
            return None
        if _SENTENCE_START_RE.match(line):
            return None
        if line.endswith(".") or line.endswith("?"):
            return None
        if not _BARE_START_RE.match(line):
            return None
        lines.append(line)

    if not lines:
        return None
    if len(lines) > 20:
        return None
    return "\n".join(lines)


def parse_action_from_text(text: str) -> Optional[ParsedAction]:
    """Extract ONE fenced code block.

    Orchestrator executes exactly one block.
    """
    if not text:
        return None
    m = CODE_BLOCK_RE.search(text)
    if m:
        lang_raw = (m.group(1) or "").strip()
        lang = _normalize_lang(lang_raw)
        if not lang:
            debug_event("LLM_PARSE", f"unsupported_lang:{lang_raw}")
            return None
        script = (m.group(2) or "").strip()
        if not script:
            return None
        return ParsedAction(language=lang, script=script)

    bare = _parse_bare_script(text)
    if bare:
        debug_event("LLM_PARSE", "bare_script")
        return ParsedAction(language="bash", script=bare)

    debug_event("LLM_PARSE", "no_code_block")
    return None
