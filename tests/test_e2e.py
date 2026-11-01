#!/usr/bin/env python3
"""End-to-end local tests against a stub OpenAI-compatible backend."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(__file__).resolve().parent / "sample_articles"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def http_json(url: str, *, method="GET", headers=None, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), json.loads(raw.decode() or "null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            parsed = json.loads(raw.decode() or "null")
        except Exception:
            parsed = {"_raw": raw.decode(errors="replace")}
        return e.code, dict(e.headers), parsed


def wait_http(url: str, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            status, _, body = http_json(url)
            if status == 200:
                return body
            last = (status, body)
        except Exception as exc:
            last = exc
        time.sleep(0.3)
    raise RuntimeError(f"timeout waiting for {url}: {last}")


def main() -> int:
    results = []
    stub_log = (LOG_DIR / "stub.log").open("w")
    server_log = (LOG_DIR / "server.log").open("w")
    report_path = LOG_DIR / "test_report.json"

    work = Path(tempfile.mkdtemp(prefix="kb-mcp-test-"))
    source = work / "articles"
    data = work / "data"
    source.mkdir()
    data.mkdir()
    # Copy samples
    for p in SAMPLES.iterdir():
        shutil.copy2(p, source / p.name)

    token = "test-token-" + ("x" * 40)  # >=32 chars
    env = os.environ.copy()
    env.update(
        {
            "ROUTER_BASE_URL": "http://127.0.0.1:18080/v1",
            "ROUTER_API_KEY": "stub",
            "CHAT_MODEL": "stub-chat",
            "EMBED_MODEL": "stub-embed",
            "MCP_TOKENS": f"tester:{token}",
            "MCP_ALLOWED_HOSTS": "127.0.0.1,localhost",
            "KB_SOURCE_DIR": str(source),
            "KB_DATA_DIR": str(data),
            "KB_SYNC_INTERVAL": "3600",
            "PYTHONPATH": str(ROOT),
            "LOG_LEVEL": "INFO",
        }
    )

    stub = subprocess.Popen(
        [sys.executable, str(ROOT / "tests" / "stub_backend.py"), "18080"],
        stdout=stub_log,
        stderr=subprocess.STDOUT,
    )
    server = None
    try:
        time.sleep(0.5)
        # Probe stub
        wait_http("http://127.0.0.1:18080/v1/models")
        results.append({"step": "stub_up", "ok": True})

        server = subprocess.Popen(
            [sys.executable, str(ROOT / "server.py")],
            cwd=str(ROOT),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        health = wait_http("http://127.0.0.1:8000/healthz", timeout=90)
        results.append({"step": "healthz", "ok": health.get("status") == "ok", "body": health})
        assert health.get("status") == "ok", health
        assert health.get("chunks", 0) > 0, health

        # 401 without token
        status, headers, body = http_json(
            "http://127.0.0.1:8000/mcp",
            method="POST",
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
                "host": "127.0.0.1:8000",
            },
            body={},
        )
        results.append({"step": "no_token", "status": status, "ok": status == 401, "body": body})
        assert status == 401, (status, body)

        # 421 bad host (with valid token)
        status, headers, body = http_json(
            "http://127.0.0.1:8000/mcp",
            method="POST",
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
                "authorization": f"Bearer {token}",
                "host": "evil.example.com",
            },
            body={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        # urllib may not let us override Host easily on all platforms; record whatever we get
        results.append(
            {
                "step": "bad_host",
                "status": status,
                "ok": status == 421,
                "body": body if isinstance(body, dict) else str(body)[:200],
            }
        )
        assert status == 421, (status, body)

        def mcp_call(method: str, params: dict | None = None, id_=1):
            payload = {"jsonrpc": "2.0", "id": id_, "method": method}
            if params is not None:
                payload["params"] = params
            return http_json(
                "http://127.0.0.1:8000/mcp",
                method="POST",
                headers={
                    "content-type": "application/json",
                    "accept": "application/json, text/event-stream",
                    "authorization": f"Bearer {token}",
                    "host": "127.0.0.1:8000",
                },
                body=payload,
                timeout=60,
            )

        # initialize
        status, _, body = mcp_call(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
        results.append({"step": "initialize", "status": status, "ok": status == 200, "body_keys": list(body) if isinstance(body, dict) else type(body).__name__})
        assert status == 200, body

        # tools/list
        status, _, body = mcp_call("tools/list", {})
        tool_names = []
        if isinstance(body, dict):
            tools = (body.get("result") or {}).get("tools") or []
            tool_names = [t.get("name") for t in tools]
        results.append({"step": "tools_list", "status": status, "ok": status == 200, "tools": tool_names})
        assert status == 200, body
        for required in ("search_knowledge", "ask_knowledge_base", "get_article", "list_articles"):
            assert required in tool_names, tool_names

        def call_tool(name: str, arguments: dict):
            return mcp_call(
                "tools/call",
                {"name": name, "arguments": arguments},
                id_=10,
            )

        # list_articles
        status, _, body = call_tool("list_articles", {"contains": "", "limit": 20})
        results.append({"step": "list_articles", "status": status, "ok": status == 200, "body": _summarize_tool(body)})
        assert status == 200, body

        # search_knowledge
        status, _, body = call_tool("search_knowledge", {"query": "how long do refunds take", "top_k": 3})
        results.append({"step": "search_knowledge", "status": status, "ok": status == 200, "body": _summarize_tool(body)})
        assert status == 200, body

        # ask_knowledge_base
        status, _, body = call_tool("ask_knowledge_base", {"question": "How long do refunds take?", "top_k": 4})
        results.append({"step": "ask_knowledge_base", "status": status, "ok": status == 200, "body": _summarize_tool(body)})
        assert status == 200, body

        # get_article
        status, _, body = call_tool("get_article", {"source": "refunds.md"})
        results.append({"step": "get_article", "status": status, "ok": status == 200, "body": _summarize_tool(body)})
        assert status == 200, body

        # Confirm md/docx/pdf ingested (from health/manifest)
        manifest = json.loads((data / "manifest.json").read_text())
        sources = sorted(manifest.get("files", {}).keys())
        results.append(
            {
                "step": "ingestion",
                "ok": all(s in sources for s in ("refunds.md", "onboarding.md", "shipping.docx", "hours.pdf")),
                "sources": sources,
            }
        )
        assert "refunds.md" in sources
        assert "shipping.docx" in sources
        assert "hours.pdf" in sources

        print(json.dumps({"ok": True, "results": results}, indent=2))
        report_path.write_text(json.dumps({"ok": True, "results": results}, indent=2))
        return 0
    except Exception as exc:
        results.append({"step": "fatal", "ok": False, "error": str(exc)})
        print(json.dumps({"ok": False, "results": results}, indent=2))
        report_path.write_text(json.dumps({"ok": False, "results": results}, indent=2))
        # Dump tail of server log
        server_log.flush()
        try:
            print("--- server.log tail ---")
            print(Path(LOG_DIR / "server.log").read_text()[-4000:])
        except Exception:
            pass
        return 1
    finally:
        for proc in (server, stub):
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        stub_log.close()
        server_log.close()
        shutil.rmtree(work, ignore_errors=True)


def _summarize_tool(body):
    if not isinstance(body, dict):
        return str(body)[:300]
    result = body.get("result") or {}
    # Prefer structuredContent
    sc = result.get("structuredContent") or result.get("structured_content")
    if sc is not None:
        return sc
    content = result.get("content")
    if content:
        return content
    if "error" in body:
        return {"error": body["error"]}
    return {k: body[k] for k in list(body)[:6]}


if __name__ == "__main__":
    sys.exit(main())
