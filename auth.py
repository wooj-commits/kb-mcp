"""Bearer token ASGI middleware with constant-time compare and per-token rate limits."""
from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from starlette.responses import JSONResponse


class BearerAuth:
    """Reject missing/wrong tokens before the request reaches the MCP app.

    Tokens are never logged; only the token *name* is used as the rate-limit key.
    """

    def __init__(
        self,
        app,
        tokens: dict[str, str],
        per_minute: int = 60,
        open_paths: frozenset[str] = frozenset({"/healthz"}),
    ):
        self.app = app
        self.per_minute = per_minute
        self.open_paths = open_paths
        self.tokens = [(name, tok.encode()) for name, tok in tokens.items()]
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in self.open_paths:
            return await self.app(scope, receive, send)

        presented = b""
        for key, value in scope.get("headers") or []:
            if key == b"authorization" and value[:7].lower() == b"bearer ":
                presented = value[7:].strip()
                break

        caller = None
        for name, token in self.tokens:
            if hmac.compare_digest(presented, token):
                caller = name
                # keep scanning so compare_digest runs for every entry (constant work)

        if caller is None:
            return await JSONResponse(
                {"error": "invalid_token"},
                status_code=401,
                headers={"www-authenticate": 'Bearer realm="kb"'},
            )(scope, receive, send)

        window = self.hits[caller]
        now = time.monotonic()
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return await JSONResponse(
                {"error": "rate_limited"},
                status_code=429,
                headers={"retry-after": "60"},
            )(scope, receive, send)
        window.append(now)

        # Stash caller name for optional downstream logging (never the token)
        scope.setdefault("state", {})
        if isinstance(scope["state"], dict):
            scope["state"]["caller"] = caller

        await self.app(scope, receive, send)
