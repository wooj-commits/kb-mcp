"""Pydantic models returned by MCP tools (structured output)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Passage(BaseModel):
    number: int
    source: str
    title: str
    section: str
    score: float
    text: str


class SearchResult(BaseModel):
    passages: list[Passage]


class Answer(BaseModel):
    answer: str
    citations: list[Passage] = Field(default_factory=list)


class Article(BaseModel):
    source: str
    title: str
    markdown: str


class ArticleSummary(BaseModel):
    source: str
    title: str
    chunk_count: int


class ArticleList(BaseModel):
    articles: list[ArticleSummary]
