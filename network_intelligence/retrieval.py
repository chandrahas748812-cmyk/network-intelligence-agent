"""Retrieval over operational runbooks and past incidents (RAG).

A small BM25 retriever implemented in the standard library — no model
download, no GPU, deterministic. If ``sentence-transformers`` happens to
be installed it is *not* used here on purpose: the demo must run
offline with zero downloads. A dense-retrieval upgrade path is
documented in the README.

Corpus entries are runbooks and past incident reports (see ``data/``).
The agent retrieves them by incident title + signals before it starts
calling tools, so its diagnosis is grounded in written procedure and
in what fixed the same failure before.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Document:
    id: str
    kind: str  # "runbook" | "past_incident"
    title: str
    text: str

    def searchable(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass
class RetrievalHit:
    doc: Document
    score: float

    def to_dict(self) -> dict:
        return {"id": self.doc.id, "kind": self.doc.kind, "title": self.doc.title, "score": round(self.score, 3)}


class BM25Retriever:
    """Classic BM25 (k1=1.5, b=0.75) over an in-memory corpus."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self._doc_tokens: list[list[str]] = [_tokens(d.searchable()) for d in docs]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = sum(self._doc_len) / max(1, len(self._doc_len))
        self._df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(docs)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        q_terms = _tokens(query)
        if not q_terms or not self.docs:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[float, Document]] = []
        for doc, toks, dl in zip(self.docs, self._doc_tokens, self._doc_len):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in set(q_terms):
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + k1 * (1 - b + b * dl / max(1, self._avg_len))
                score += self._idf(term) * f * (k1 + 1) / denom
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievalHit(doc, s) for s, doc in scored[:top_k] if s > 0]


def load_corpus(data_dir: str | Path) -> list[Document]:
    """Load runbooks + past incidents from ``data/`` into Documents."""
    data_dir = Path(data_dir)
    docs: list[Document] = []
    runbooks_path = data_dir / "runbooks.json"
    if runbooks_path.exists():
        for rb in json.loads(runbooks_path.read_text()):
            text = (
                f"Symptoms: {'; '.join(rb.get('symptoms', []))}\n"
                f"Checks: {'; '.join(rb.get('checks', []))}\n"
                f"Common causes: {'; '.join(rb.get('common_causes', []))}\n"
                f"Remediation: {rb.get('remediation', '')}"
            )
            docs.append(Document(rb["id"], "runbook", rb["title"], text))
    incidents_path = data_dir / "incidents.json"
    if incidents_path.exists():
        for inc in json.loads(incidents_path.read_text()):
            text = (
                f"Summary: {inc.get('summary', '')}\n"
                f"Root cause: {inc.get('root_cause', '')}\n"
                f"Resolution: {inc.get('resolution', '')}"
            )
            docs.append(Document(inc["id"], "past_incident", inc["title"], text))
    return docs


def default_retriever(data_dir: str | Path | None = None) -> BM25Retriever:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    return BM25Retriever(load_corpus(data_dir))
