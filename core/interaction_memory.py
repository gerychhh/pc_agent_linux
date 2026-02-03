from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


MEMORY_PATH = PROJECT_ROOT / "interaction_memory.json"
MAX_HISTORY = 200


def _default_memory() -> dict[str, Any]:
    return {"routes": {}, "actions": {}, "history": []}


def _norm(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def load_memory() -> dict[str, Any]:
    if not MEMORY_PATH.exists():
        return _default_memory()
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_memory()
    if not isinstance(data, dict):
        return _default_memory()
    if "routes" not in data or not isinstance(data.get("routes"), dict):
        data["routes"] = {}
    if "actions" not in data or not isinstance(data.get("actions"), dict):
        data["actions"] = {}
    if "history" not in data or not isinstance(data.get("history"), list):
        data["history"] = []
    return data


def save_memory(memory: dict[str, Any]) -> None:
    tmp_path = Path(f"{MEMORY_PATH}.tmp")
    tmp_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(MEMORY_PATH)


# ---------- Short-hands (routes) ----------

def get_route(query: str) -> str | None:
    if not query:
        return None
    memory = load_memory()
    route = memory.get("routes", {}).get(_norm(query))
    return route if isinstance(route, str) and route else None


def set_route(query: str, corrected_query: str) -> None:
    if not query or not corrected_query:
        return
    memory = load_memory()
    routes = memory.setdefault("routes", {})
    routes[_norm(query)] = corrected_query
    save_memory(memory)


def delete_route(query: str) -> None:
    if not query:
        return
    memory = load_memory()
    routes = memory.get("routes", {})
    routes.pop(_norm(query), None)
    save_memory(memory)


def find_similar_routes(query: str, limit: int = 3) -> list[dict[str, Any]]:
    if not query:
        return []
    memory = load_memory()
    routes = memory.get("routes", {})
    q = _norm(query)
    results: list[dict[str, Any]] = []
    for stored_query, resolved in routes.items():
        if not isinstance(stored_query, str) or not isinstance(resolved, str):
            continue
        score = SequenceMatcher(None, q, _norm(stored_query)).ratio()
        results.append({"query": stored_query, "resolved": resolved, "score": score})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


# ---------- Learned actions (no LLM next time) ----------

def get_action(query: str) -> dict[str, Any] | None:
    if not query:
        return None
    memory = load_memory()
    action = memory.get("actions", {}).get(_norm(query))
    return action if isinstance(action, dict) and action else None


def set_action(query: str, action: dict[str, Any]) -> None:
    if not query or not isinstance(action, dict) or not action:
        return
    memory = load_memory()
    actions = memory.setdefault("actions", {})
    actions[_norm(query)] = action
    save_memory(memory)


def delete_action(query: str) -> None:
    if not query:
        return
    memory = load_memory()
    actions = memory.get("actions", {})
    actions.pop(_norm(query), None)
    save_memory(memory)


def find_similar_actions(query: str, limit: int = 3) -> list[dict[str, Any]]:
    if not query:
        return []
    memory = load_memory()
    actions = memory.get("actions", {})
    q = _norm(query)
    results: list[dict[str, Any]] = []
    for stored_query, action in actions.items():
        if not isinstance(stored_query, str) or not isinstance(action, dict):
            continue
        score = SequenceMatcher(None, q, _norm(stored_query)).ratio()
        results.append({"query": stored_query, "action": action, "score": score})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


# ---------- History ----------

def record_history(query: str, response: str, resolved_query: str) -> None:
    if not query or not response:
        return
    memory = load_memory()
    history = memory.setdefault("history", [])
    history.append(
        {
            "query": query,
            "resolved_query": resolved_query,
            "response": response,
            "timestamp": int(time.time()),
        }
    )
    if len(history) > MAX_HISTORY:
        memory["history"] = history[-MAX_HISTORY:]
    save_memory(memory)
