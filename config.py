"""Environment configuration for the knowledge-base MCP server."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"refusing to start: {name} is empty")
    return value


def load_tokens(raw: str | None = None) -> dict[str, str]:
    """Parse MCP_TOKENS as comma-separated name:token pairs.

    Refuses to start when empty or any token is shorter than 32 characters.
    """
    text = (raw if raw is not None else os.environ.get("MCP_TOKENS", "")).strip()
    if not text:
        raise SystemExit("refusing to start: MCP_TOKENS is empty")
    tokens: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"refusing to start: MCP_TOKENS entry {part!r} is not name:token"
            )
        name, token = part.split(":", 1)
        name, token = name.strip(), token.strip()
        if not name or not token:
            raise SystemExit("refusing to start: MCP_TOKENS has an empty name or token")
        if len(token) < 32:
            raise SystemExit(
                f"refusing to start: token for {name!r} is shorter than 32 characters"
            )
        if name in tokens:
            raise SystemExit(f"refusing to start: duplicate token name {name!r}")
        tokens[name] = token
    if not tokens:
        raise SystemExit("refusing to start: MCP_TOKENS is empty")
    return tokens


@dataclass(frozen=True)
class Settings:
    router_base_url: str
    router_api_key: str
    chat_model: str
    embed_model: str
    tokens: dict[str, str]
    allowed_hosts: list[str]
    source_dir: str
    data_dir: str
    sync_interval_seconds: int = 900
    rate_limit_per_minute: int = 60
    embed_batch_size: int = 32
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_tokens: int = 800
    temperature: float = 0.2


def load_settings() -> Settings:
    hosts_raw = os.environ.get("MCP_ALLOWED_HOSTS", "").strip()
    if not hosts_raw:
        # Allow localhost for local testing when explicitly set via default below
        # Production must set MCP_ALLOWED_HOSTS.
        raise SystemExit("refusing to start: MCP_ALLOWED_HOSTS is empty")
    hosts = [h.strip() for h in hosts_raw.split(",") if h.strip()]
    # Also accept host:port variants for local uvicorn
    expanded: list[str] = []
    for h in hosts:
        expanded.append(h)
        if ":" not in h and not h.endswith(":*"):
            expanded.append(f"{h}:*")
    return Settings(
        router_base_url=_require("ROUTER_BASE_URL").rstrip("/"),
        router_api_key=os.environ.get("ROUTER_API_KEY", "olares").strip() or "olares",
        chat_model=os.environ.get("CHAT_MODEL", "default-chat").strip() or "default-chat",
        embed_model=_require("EMBED_MODEL"),
        tokens=load_tokens(),
        allowed_hosts=expanded,
        source_dir=os.environ.get("KB_SOURCE_DIR", "/kb/articles").strip() or "/kb/articles",
        data_dir=os.environ.get("KB_DATA_DIR", "/app/data").strip() or "/app/data",
        sync_interval_seconds=int(os.environ.get("KB_SYNC_INTERVAL", "900")),
        rate_limit_per_minute=int(os.environ.get("MCP_RATE_LIMIT", "60")),
    )


def log_info(msg: str, *args) -> None:
    print(msg % args if args else msg, file=sys.stderr, flush=True)
