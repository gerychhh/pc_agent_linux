from __future__ import annotations

"""core.window_manager (Linux-only)

We deliberately removed all Windows-specific logic.

Best-effort active window info on Linux/X11:
- uses xprop (_NET_ACTIVE_WINDOW / _NET_WM_PID / _NET_WM_NAME)
If xprop is missing or you're on Wayland without XWayland, returns empty fields.
"""

from typing import Any
import re
import subprocess

import psutil


def _safe_process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except Exception:
        return ""


def get_active_window_info() -> dict[str, Any]:
    try:
        active = subprocess.check_output(
            ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "ignore")
        wid = active.split()[-1].strip()
        if wid == "0x0":
            return {"title": "", "hwnd": 0, "pid": 0, "process": ""}

        pid_out = subprocess.check_output(
            ["xprop", "-id", wid, "_NET_WM_PID"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "ignore")
        pid = 0
        pm = re.search(r"=\s*(\d+)", pid_out)
        if pm:
            pid = int(pm.group(1))

        title_out = subprocess.check_output(
            ["xprop", "-id", wid, "_NET_WM_NAME"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "ignore")
        tm = re.search(r'=\s*"(.*)"\s*$', title_out.strip())
        title = tm.group(1) if tm else ""
        return {"title": title, "hwnd": wid, "pid": pid, "process": _safe_process_name(pid)}
    except Exception:
        return {"title": "", "hwnd": 0, "pid": 0, "process": ""}


def close_window_by_pid(pid: int) -> None:
    try:
        psutil.Process(int(pid)).terminate()
    except Exception:
        pass