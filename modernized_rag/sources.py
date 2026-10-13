"""Load the original URL corpus as clean source documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .backend import SourceDocument


def read_urls(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def fetch_documents(urls: Iterable[str], timeout_seconds: int = 15) -> list[SourceDocument]:
    import requests
    from bs4 import BeautifulSoup

    documents: list[SourceDocument] = []
    headers = {"User-Agent": "FITPT educational RAG prototype/2.0"}
    for url in urls:
        response = requests.get(url, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "form"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = " ".join(soup.get_text(" ", strip=True).split())
        if len(text.split()) >= 100:
            documents.append(SourceDocument(source=url, title=title, text=text))
    return documents
