# kb-mcp

Private knowledge-base MCP server for [Olares](https://www.olares.com/) One.

- Streamable HTTP at `/mcp` (stateless JSON)
- Bearer tokens (`MCP_TOKENS`), open `/healthz`
- Tools: `search_knowledge`, `ask_knowledge_base`, `get_article`, `list_articles`
- Chroma index + MarkItDown ingestion; chat/embeddings via Olares Router

## Image

```
ghcr.io/wooj-commits/kb-mcp:0.1.0
ghcr.io/wooj-commits/kb-mcp:latest
```

Built by GitHub Actions on push to `main` (linux/amd64). The GHCR package must be **Public** so Olares can pull without a pull secret.

## Local test

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_e2e.py
```

## Olares chart

Key-less template: `chart/kbmcp/`. Bake tokens with `build-package.sh` only on a secure machine; never commit tokens.
