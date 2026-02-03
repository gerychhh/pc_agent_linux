from __future__ import annotations

import os
import re


def validate_python(script: str) -> list[str]:
    """Basic guardrails for python snippets.

    This is not a sandbox; it's just a cheap safety net for obvious destructive ops.
    """
    errors: list[str] = []

    if re.search(r"\b(os\.remove|os\.rmdir|shutil\.rmtree|Path\.unlink)\b", script):
        errors.append("Опасное удаление файлов/папок запрещено.")

    # discourage raw writing of office formats as plain text
    if re.search(r"open\([^\n]*\.(docx|xlsx|pdf|pptx)[^\n]*['\"]w", script, re.IGNORECASE):
        errors.append("Нельзя создавать docx/xlsx/pptx/pdf через open(...,'w').")

    if ("Path(" in script or "Path." in script) and "from pathlib import Path" not in script:
        errors.append("Если используется Path, нужен импорт: from pathlib import Path.")

    if re.search(r"\.docx", script, re.IGNORECASE):
        if "from docx import Document" not in script:
            errors.append("Для .docx нужен python-docx: from docx import Document.")

    return errors


def validate_bash(script: str) -> list[str]:
    """Basic guardrails for bash scripts.

    Blocks obviously dangerous commands:
    - rm -rf
    - mkfs / fdisk / parted / dd
    - sudo (unless PC_AGENT_ALLOW_SUDO=1)

    NOTE: This is NOT a sandbox. It's only a safety net.
    """
    errors: list[str] = []
    lowered = script.lower()

    if re.search(r"\brm\s+-rf\b", lowered):
        errors.append("Опасная команда: rm -rf запрещена.")
    if re.search(r"\bmkfs\b|\bparted\b|\bfdisk\b|\bgdisk\b|\bsfdisk\b", lowered):
        errors.append("Команды работы с разделами/файловыми системами запрещены.")
    if re.search(r"\bdd\b", lowered):
        errors.append("Опасная команда: dd запрещена.")

    allow_sudo = os.getenv("PC_AGENT_ALLOW_SUDO", "0").strip().lower() in {"1", "true", "yes", "on"}
    if re.search(r"\bsudo\b", lowered) and not allow_sudo:
        errors.append("sudo запрещён в автоматическом выполнении. Если ты понимаешь риски: PC_AGENT_ALLOW_SUDO=1")

    # also block pkexec by default (similar risk level)
    allow_pkexec = os.getenv("PC_AGENT_ALLOW_PKEXEC", "0").strip().lower() in {"1", "true", "yes", "on"}
    if re.search(r"\bpkexec\b", lowered) and not allow_pkexec:
        errors.append("pkexec запрещён в автоматическом выполнении. Если нужно: PC_AGENT_ALLOW_PKEXEC=1")

    return errors
