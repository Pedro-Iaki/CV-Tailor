"""Find, start and check Ollama, and download models, for the desktop app's setup screen.

Ollama itself can't be portable (it installs into the user profile), so the app
guides the user: running? installed but stopped? not installed (link to the
official download, or winget)? Then: are the recommended models downloaded?
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

DOWNLOAD_URL = "https://ollama.com/download"
WINGET_ID = "Ollama.Ollama"

# (model, approx. download, what it is for), from the 2026-11-17 benchmark and ratings
# on an 8 GB GPU. Only Qwen3.5 made CVs worth sending; the rest were dropped.
MAIN_MODEL = "qwen3.5:35b-a3b"
RECOMMENDED = [
    (MAIN_MODEL, "22 GB", "the model CV Tailor uses; 1-3 min per CV, 6-10 min careful"),
    ("bge-m3", "1.2 GB", "optional: relevance hints in careful mode"),
]
# Thorough mode thinks in its free-text passes, which needs a longer context than
# Ollama's default. This is the main model under another name with a 16k window: no
# extra download, it reuses the same weights.
LONG_CONTEXT_MODEL = "cvtailor-qwen3.5-16k"
LONG_CONTEXT = 16384


def server_url(base_url: Optional[str]) -> str:
    """http://host:port of the Ollama server from an OpenAI-style base URL."""
    return re.sub(r"/v1/?$", "", (base_url or "http://localhost:11434/v1").rstrip("/"))


def _get(url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def find_executable() -> Optional[str]:
    """ollama.exe (CLI), or the tray app, if installed."""
    found = shutil.which("ollama")
    if found:
        return found
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama"
        for name in ("ollama.exe", "ollama app.exe"):
            if (base / name).exists():
                return str(base / name)
    for p in ("/usr/local/bin/ollama", "/Applications/Ollama.app/Contents/Resources/ollama"):
        if Path(p).exists():
            return p
    return None


def status(base_url: Optional[str]) -> dict:
    """{running, version, installed, executable, models: [names], winget}."""
    srv = server_url(base_url)
    ver = _get(srv + "/api/version")
    tags = _get(srv + "/api/tags") if ver else None
    exe = find_executable()
    return {
        "server": srv,
        "running": ver is not None,
        "version": (ver or {}).get("version"),
        "installed": exe is not None or ver is not None,
        "executable": exe,
        "models": sorted(m.get("name", "") for m in (tags or {}).get("models", [])),
        "winget": shutil.which("winget") is not None,
        "download_url": DOWNLOAD_URL,
    }


def has_model(models: list[str], name: str) -> bool:
    want = name if ":" in name else name + ":latest"
    return want in models or name in models


def start(executable: Optional[str] = None) -> bool:
    """Start Ollama in the background (the tray app if present, else `ollama serve`).
    Returns False if it isn't installed."""
    exe = executable or find_executable()
    if not exe:
        return False
    flags = 0
    if sys.platform.startswith("win"):
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        tray = Path(exe).with_name("ollama app.exe")
        if tray.exists():
            subprocess.Popen([str(tray)], creationflags=flags, close_fds=True)
            return True
    subprocess.Popen([exe, "serve"], creationflags=flags, close_fds=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def install_with_winget() -> bool:
    """Open a console running winget to install Ollama (the user confirms there)."""
    if not shutil.which("winget"):
        return False
    subprocess.Popen(["cmd", "/c", "start", "Install Ollama", "winget", "install", "-e", "--id", WINGET_ID])
    return True


def pull(base_url: Optional[str], model: str, on_progress: Callable[[str, int, int], None]) -> None:
    """Download a model, reporting (status, completed bytes, total bytes)."""
    req = urllib.request.Request(server_url(base_url) + "/api/pull",
                                 data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("error"):
                raise RuntimeError(msg["error"])
            on_progress(msg.get("status", ""), int(msg.get("completed") or 0), int(msg.get("total") or 0))


def create_long_context(base_url: Optional[str], source: str = MAIN_MODEL,
                        name: str = LONG_CONTEXT_MODEL, num_ctx: int = LONG_CONTEXT) -> str:
    """Create `name`: `source` with a larger context window (no download; same weights)."""
    body = {"model": name, "from": source, "parameters": {"num_ctx": num_ctx}, "stream": False}
    req = urllib.request.Request(server_url(base_url) + "/api/create", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        msg = json.loads(resp.read().decode("utf-8") or "{}")
    if msg.get("error"):
        raise RuntimeError(msg["error"])
    return name
