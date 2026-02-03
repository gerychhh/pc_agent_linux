from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass

from .debug import debug_event


@dataclass
class ParsedAction:
    language: str
    script: str


CODE_BLOCK_RE = re.compile(r"```(python|bash|sh|json)\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_action_from_text(text: str) -> Optional[ParsedAction]:
    """Extract ONE fenced code block.

    Orchestrator executes exactly one block.
    """
    if not text:
        return None
    m = CODE_BLOCK_RE.search(text)
    if not m:
        debug_event("LLM_PARSE", "no_code_block")
        return None
    lang = (m.group(1) or "").lower().strip()
    script = (m.group(2) or "").strip()
    if lang == "sh":
        lang = "bash"
    if not script:
        return None
    return ParsedAction(language=lang, script=script)
