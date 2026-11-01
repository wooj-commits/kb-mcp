"""Convert, chunk, embed and sync knowledge articles into Chroma."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from config import Settings, log_info

logger = logging.getLogger("kb.ingest")

SKIP_PREFIXES = (".", "~$", "~")
CONVERT_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}
TEXT_EXTS = {".md", ".markdown", ".txt"}
ALL_EXTS = CONVERT_EXTS | TEXT_EXTS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_source_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_skip(path: Path) -> bool:
    name = path.name
    if name.startswith(".") or name.startswith("~$"):
        return True
    if path.suffix.lower() not in ALL_EXTS:
        return True
    return False


def convert_to_markdown(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace")
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(path))
    text = (result.text_content or "").strip()
    if not text and ext == ".pdf":
        raise ValueError(
            f"{path.name} yielded no text (likely a scan); convert with Docling first"
        )
    return text


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _ensure_title(markdown: str, fallback_title: str) -> tuple[str, str]:
    """Ensure the doc starts with a # title; return (title, markdown)."""
    m = re.match(r"^#\s+(.+?)\s*$", markdown.lstrip(), re.MULTILINE)
    if m:
        return m.group(1).strip(), markdown.lstrip()
    titled = f"# {fallback_title}\n\n{markdown.lstrip()}"
    return fallback_title, titled


def chunk_markdown(
    markdown: str,
    *,
    fallback_title: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> tuple[str, list[tuple[str, str]]]:
    """Split on headings, then pack paragraphs to ~chunk_size with overlap.

    Returns (title, [(section_path, chunk_text), ...]).
    Each chunk text includes its heading line.
    """
    title, markdown = _ensure_title(markdown, fallback_title)

    # Split into (heading_path, body) sections
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        sections.append((title, markdown.strip()))
    else:
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            path = " > ".join(h for _, h in stack)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            body = markdown[start:end].strip()
            heading_line = m.group(0).strip()
            sections.append((path, f"{heading_line}\n\n{body}".strip() if body else heading_line))

    chunks: list[tuple[str, str]] = []
    for section_path, text in sections:
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append((section_path, text.strip()))
            continue
        # Pack by paragraphs
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for para in paras:
            candidate = f"{buf}\n\n{para}".strip() if buf else para
            if len(candidate) <= chunk_size:
                buf = candidate
                continue
            if buf:
                chunks.append((section_path, buf))
                # overlap from end of buf
                if overlap > 0 and len(buf) > overlap:
                    tail = buf[-overlap:]
                    # start at a whitespace boundary when possible
                    sp = tail.find(" ")
                    if 0 < sp < len(tail) - 1:
                        tail = tail[sp + 1 :]
                    buf = f"{tail}\n\n{para}".strip()
                else:
                    buf = para
            else:
                # single paragraph longer than chunk_size: hard split
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    piece = para[start:end]
                    chunks.append((section_path, piece))
                    if end >= len(para):
                        buf = ""
                        break
                    start = max(end - overlap, start + 1)
                    buf = ""
        if buf.strip():
            chunks.append((section_path, buf.strip()))
    return title, chunks


def passage_embed_text(title: str, chunk: str) -> str:
    t = title.strip() if title.strip() else "none"
    return f"title: {t} | text: {chunk}"


def query_embed_text(question: str) -> str:
    return f"task: search result | query: {question}"


class KnowledgeIndex:
    """Chroma-backed index with a SHA-256 manifest for sync."""

    def __init__(self, settings: Settings, llm: OpenAI):
        self.settings = settings
        self.llm = llm
        self.data_dir = Path(settings.data_dir)
        self.source_dir = Path(settings.source_dir)
        self.index_dir = self.data_dir / "index"
        self.articles_dir = self.data_dir / "articles"
        self.manifest_path = self.data_dir / "manifest.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir.mkdir(parents=True, exist_ok=True)

        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.index_dir))
        # Prefer modern configuration API; fall back to metadata hnsw:space
        try:
            self.collection = self._client.get_or_create_collection(
                name="knowledge",
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except TypeError:
            self.collection = self._client.get_or_create_collection(
                name="knowledge",
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Already exists with different config — open it
            self.collection = self._client.get_or_create_collection(
                name="knowledge",
                embedding_function=None,
            )
        self._manifest: dict[str, Any] = self._load_manifest()
        self._lock = threading.RLock()

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("manifest unreadable (%s); starting fresh", exc)
        return {"version": 1, "files": {}}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[dict[str, Any]]:
        files = self._manifest.get("files", {})
        return [
            {
                "source": src,
                "title": meta.get("title", src),
                "chunk_count": int(meta.get("chunk_count", 0)),
            }
            for src, meta in sorted(files.items())
        ]

    def get_article_markdown(self, source: str) -> tuple[str, str] | None:
        """Return (title, markdown) or None."""
        meta = self._manifest.get("files", {}).get(source)
        safe = source.replace("/", "__")
        path = self.articles_dir / f"{safe}.md"
        if not path.exists():
            return None
        title = (meta or {}).get("title") or Path(source).stem
        return title, path.read_text(encoding="utf-8", errors="replace")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch = self.settings.embed_batch_size
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = self.llm.embeddings.create(model=self.settings.embed_model, input=chunk)
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)
        return vectors

    def _delete_source(self, source: str) -> None:
        try:
            self.collection.delete(where={"source": source})
        except Exception as exc:
            logger.warning("delete chunks for %s: %s", source, exc)
        safe = source.replace("/", "__")
        art = self.articles_dir / f"{safe}.md"
        if art.exists():
            art.unlink()
        self._manifest.get("files", {}).pop(source, None)

    def _ingest_file(self, path: Path, source: str, file_hash: str) -> None:
        markdown = convert_to_markdown(path)
        title, chunks = chunk_markdown(
            markdown,
            fallback_title=path.stem,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            logger.warning("no chunks from %s; skipping", source)
            return

        texts = [passage_embed_text(title, c) for _, c in chunks]
        vectors = self._embed_batch(texts)
        ids = [f"{source}#{i}" for i in range(len(chunks))]
        documents = [c for _, c in chunks]
        metadatas = [
            {
                "source": source,
                "title": title,
                "section": section,
                "chunk": i,
            }
            for i, (section, _) in enumerate(chunks)
        ]

        self._delete_source(source)
        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        safe = source.replace("/", "__")
        (self.articles_dir / f"{safe}.md").write_text(markdown, encoding="utf-8")
        self._manifest.setdefault("files", {})[source] = {
            "sha256": file_hash,
            "title": title,
            "chunk_count": len(chunks),
            "mtime": path.stat().st_mtime,
        }

    def sync_once(self) -> dict[str, int]:
        """Sync source_dir into the index. Returns counts."""
        with self._lock:
            stats = {"unchanged": 0, "updated": 0, "removed": 0, "failed": 0}
            if not self.source_dir.exists():
                log_info("source dir %s missing; skipping sync", self.source_dir)
                return stats

            seen: set[str] = set()
            for path in sorted(self.source_dir.rglob("*")):
                if not path.is_file() or should_skip(path):
                    continue
                try:
                    source = relative_source_id(path, self.source_dir)
                    seen.add(source)
                    file_hash = sha256_file(path)
                    prev = self._manifest.get("files", {}).get(source)
                    if prev and prev.get("sha256") == file_hash:
                        stats["unchanged"] += 1
                        continue
                    self._ingest_file(path, source, file_hash)
                    stats["updated"] += 1
                    log_info("indexed %s (%s chunks)", source,
                             self._manifest["files"][source]["chunk_count"])
                except Exception as exc:
                    stats["failed"] += 1
                    logger.exception("failed to ingest %s: %s", path, exc)

            for source in list(self._manifest.get("files", {}).keys()):
                if source not in seen:
                    self._delete_source(source)
                    stats["removed"] += 1
                    log_info("removed %s", source)

            self._save_manifest()
            return stats

    def start_background_sync(self, interval: int | None = None) -> threading.Thread:
        interval = interval if interval is not None else self.settings.sync_interval_seconds

        def loop():
            while True:
                time.sleep(interval)
                try:
                    stats = self.sync_once()
                    log_info("periodic sync: %s", stats)
                except Exception:
                    logger.exception("periodic sync failed")

        t = threading.Thread(target=loop, name="kb-sync", daemon=True)
        t.start()
        return t
