from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from .config import PROJECT_ROOT
from .state import add_recent_file, add_recent_url, set_active_file, set_active_url


SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _result(ok: bool, **kwargs: Any) -> dict[str, Any]:
    payload = {"ok": ok, **kwargs}
    if ok:
        payload.setdefault("stdout", "")
        payload.setdefault("stderr", "")
        payload.setdefault("returncode", 0)
    else:
        payload.setdefault("stdout", "")
        payload.setdefault("stderr", "")
        payload.setdefault("returncode", 1)
    return payload


def _run_command(cmd: Sequence[str], script_path: Path, timeout_sec: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        duration_ms = int((time.time() - start) * 1000)
        out = _result(
            proc.returncode == 0,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            returncode=proc.returncode,
            duration_ms=duration_ms,
            script_path=str(script_path),
            exec_cmd=" ".join(cmd),
        )
        return out
    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start) * 1000)
        return _result(False, error="timeout", duration_ms=duration_ms, script_path=str(script_path), exec_cmd=" ".join(cmd))
    except Exception as exc:
        duration_ms = int((time.time() - start) * 1000)
        return _result(False, error=str(exc), duration_ms=duration_ms, script_path=str(script_path), exec_cmd=" ".join(cmd))


def _track_shell_script(code: str) -> None:
    # Track xdg-open URLs/paths (best-effort)
    for m in re.finditer(r"\bxdg-open\s+(['\"]?)([^'\"\n]+)\1", code):
        target = (m.group(2) or "").strip()
        if not target:
            continue
        if target.startswith("http://") or target.startswith("https://"):
            add_recent_url(target)
            set_active_url(target)
        else:
            p = Path(target).expanduser()
            add_recent_file(str(p))
            set_active_file(str(p))


def run_python(code: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{ts}.py"
    script_path.write_text(code, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return _run_command(["python3", str(script_path)], script_path, timeout_sec, env=env)


def run_shell(code: str, timeout_sec: int) -> dict[str, Any]:
    """Run bash script."""
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{ts}.sh"
    script_path.write_text(code, encoding="utf-8")
    _track_shell_script(code)
    return _run_command(["bash", str(script_path)], script_path, timeout_sec)


def run_pip_install(package: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{ts}_pip.sh"
    code = f"python3 -m pip install --no-input {package}\n"
    script_path.write_text(code, encoding="utf-8")
    return _run_command(["bash", str(script_path)], script_path, timeout_sec)
