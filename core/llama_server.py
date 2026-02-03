from __future__ import annotations

import atexit
import logging
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import shutil

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_SERVER_PROC: Optional[subprocess.Popen] = None


def _abs_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _ping_models(base_url: str, timeout_s: float = 0.6) -> bool:
    url = base_url.rstrip('/') + '/models'
    req = Request(url, headers={'Accept': 'application/json'})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return 200 <= getattr(resp, 'status', 200) < 300
    except Exception:
        return False


def _stop_server() -> None:
    global _SERVER_PROC
    proc = _SERVER_PROC
    _SERVER_PROC = None
    if not proc:
        return
    try:
        # Kill the whole process group (server may spawn workers)
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


atexit.register(_stop_server)



def _read_tail(path: Path, max_lines: int = 200) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    lines = txt.splitlines()
    return "\n".join(lines[-max_lines:])


def ensure_llama_server_running() -> bool:
    """Start local llama.cpp server if:
    - backend points to localhost base_url (/v1)
    - PC_AGENT_AUTOSTART_LLAMACPP is enabled (default: 1)
    """
    autostart = os.getenv('PC_AGENT_AUTOSTART_LLAMACPP', '1').strip().lower()
    if autostart in {'0', 'false', 'no', 'off'}:
        return False

    base_url = os.getenv('PC_AGENT_BASE_URL') or os.getenv('OPENAI_BASE_URL') or 'http://127.0.0.1:1234/v1'
    if not base_url.startswith(('http://127.0.0.1', 'http://localhost', 'http://0.0.0.0')):
        return False

    if _ping_models(base_url):
        return True

    return _start_server(base_url)


def _start_server(base_url: str) -> bool:
    global _SERVER_PROC

    # Parse host/port from base_url (expect ...://host:port/v1)
    u = urlparse(base_url)
    host = os.getenv('LLAMA_SERVER_HOST', u.hostname or '127.0.0.1')
    port = int(os.getenv('LLAMA_SERVER_PORT', str(u.port or 1234)))

    # Resolve binary
    bin_env = os.getenv('LLAMA_SERVER_BIN', '').strip()
    candidates = []
    if bin_env:
        candidates.append(_abs_path(bin_env))
    candidates.append((PROJECT_ROOT / 'third_party' / 'llama.cpp' / 'build' / 'bin' / 'llama-server').resolve())
    candidates.append((PROJECT_ROOT / 'third_party' / 'llama.cpp' / 'build' / 'bin' / 'llama-server.exe').resolve())
    # PATH lookup
    which = shutil.which('llama-server')
    if which:
        candidates.append(Path(which))

    server_bin = next((p for p in candidates if p.exists()), None)
    if not server_bin:
        logger.error('LLM server autostart: llama-server binary not found. Set LLAMA_SERVER_BIN or build llama.cpp into third_party/llama.cpp.')
        return False

    # Model: local file OR HuggingFace repo/file
    model_path_env = os.getenv('LLAMA_SERVER_MODEL', '').strip()
    hf_repo = os.getenv('LLAMA_SERVER_HF_REPO', '').strip()
    hf_file = os.getenv('LLAMA_SERVER_HF_FILE', '').strip()

    args = [str(server_bin), '--host', host, '--port', str(port)]

    # Make model name stable for OpenAI-compatible clients
    alias = os.getenv('PC_AGENT_MODEL_NAME', 'local-model').strip() or 'local-model'
    args += ['--alias', alias]

    # Performance / GPU
    ctx = os.getenv('LLAMA_SERVER_CTX', '4096').strip()
    ngl_env = os.getenv('LLAMA_SERVER_N_GPU_LAYERS', '999').strip()
    threads = os.getenv('LLAMA_SERVER_THREADS', str(max(1, os.cpu_count() or 8))).strip()

    # We'll try to start with the requested n_gpu_layers, and if we hit CUDA OOM
    # we'll automatically reduce layers a few times.
    try:
        ngl0 = int(ngl_env)
    except Exception:
        ngl0 = 999

    max_retries = int(os.getenv('LLAMA_SERVER_AUTOFIT_RETRIES', '6'))
    step = int(os.getenv('LLAMA_SERVER_AUTOFIT_STEP', '5'))

    
    # Keep server lean
    args += ['--no-webui']

    if model_path_env:
        model_path = _abs_path(model_path_env)
        if not model_path.exists():
            logger.error('LLM server autostart: model file not found: %s', model_path)
            return False
        args += ['--model', str(model_path)]
    elif hf_repo and hf_file:
        args += ['--hf-repo', hf_repo, '--hf-file', hf_file]
    else:
        logger.error('LLM server autostart: no model configured. Set LLAMA_SERVER_MODEL or LLAMA_SERVER_HF_REPO + LLAMA_SERVER_HF_FILE.')
        return False

    # Extra args (optional)
    extra = os.getenv('LLAMA_SERVER_EXTRA_ARGS', '').strip()
    if extra:
        args += shlex.split(extra)


    # Logs
    logs_dir = (PROJECT_ROOT / 'logs')
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / 'llama_server.log'

    # We re-open the log file each attempt (append) for clearer history.
    for attempt in range(max_retries + 1):
        ngl = max(0, ngl0 - attempt * step)

        # Rebuild args for this attempt
        args_try = list(args)
        args_try += ['--ctx-size', ctx, '--n-gpu-layers', str(ngl), '--threads', threads]

        # Keep server lean
        args_try += ['--no-webui']

        # Model: local file OR HuggingFace repo/file
        model_path_env = os.getenv('LLAMA_SERVER_MODEL', '').strip()
        hf_repo = os.getenv('LLAMA_SERVER_HF_REPO', '').strip()
        hf_file = os.getenv('LLAMA_SERVER_HF_FILE', '').strip()

        if model_path_env:
            model_path = _abs_path(model_path_env)
            if not model_path.exists():
                logger.error('LLM server autostart: model file not found: %s', model_path)
                return False
            args_try += ['--model', str(model_path)]
        elif hf_repo and hf_file:
            args_try += ['--hf-repo', hf_repo, '--hf-file', hf_file]
        else:
            logger.error('LLM server autostart: no model configured. Set LLAMA_SERVER_MODEL or LLAMA_SERVER_HF_REPO + LLAMA_SERVER_HF_FILE.')
            return False

        # Extra args (optional)
        extra = os.getenv('LLAMA_SERVER_EXTRA_ARGS', '').strip()
        if extra:
            args_try += shlex.split(extra)

        log_fh = open(log_path, 'a', encoding='utf-8', buffering=1)

        logger.info('Starting llama-server (attempt=%d n_gpu_layers=%d): %s',
                    attempt + 1, ngl, ' '.join(shlex.quote(a) for a in args_try))
        try:
            _SERVER_PROC = subprocess.Popen(
                args_try,
                stdout=log_fh,
                stderr=log_fh,
                preexec_fn=os.setsid,
                cwd=str(PROJECT_ROOT),
            )
        except Exception as e:
            logger.exception('Failed to start llama-server: %s', e)
            _SERVER_PROC = None
            return False

        # Wait until server responds
        deadline = time.monotonic() + float(os.getenv('LLAMA_SERVER_STARTUP_TIMEOUT', '20'))
        while time.monotonic() < deadline:
            if _ping_models(base_url, timeout_s=0.6):
                logger.info('llama-server is up: %s', base_url)
                return True
            time.sleep(0.4)

        # Not ready: stop and check logs for OOM
        _stop_server()
        tail = _read_tail(log_path, max_lines=160).lower()
        oom = ('out of memory' in tail) or ('cudamalloc failed' in tail) or ('failed to allocate' in tail)
        if oom:
            logger.warning('llama-server failed due to OOM, retrying with fewer GPU layers (next attempt)...')
            continue

        logger.error('llama-server did not become ready in time (not OOM). See logs: %s', log_path)
        return False

    logger.error('llama-server autostart: exhausted retries. Likely not enough VRAM. See logs: %s', log_path)
    return False

