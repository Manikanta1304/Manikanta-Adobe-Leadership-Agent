"""
Document Ingestion Module
Loads PDF, DOCX, and TXT files from a directory,
splits them into overlapping chunks, and returns a list of
chunk dicts ready for embedding.
"""

import os
import re
from typing import List, Dict
from pathlib import Path


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf:  pip install pypdf")

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _load_docx(path: str) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        import docx
    except ImportError:
        raise ImportError("Install python-docx:  pip install python-docx")

    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


LOADERS = {
    ".txt":  _load_txt,
    ".md":   _load_txt,
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
}


def _clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    # Collapse runs of spaces/tabs on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Collapse more-than-two consecutive blank lines
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result.strip()


def _chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """
    Split text into overlapping windows.

    Each chunk is a dict:
        { "text": str, "source": str, "chunk_id": int }
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_id": chunk_id,
        })
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
        chunk_id += 1

    return chunks


class DocumentIngester:
    """
    Walks a directory, loads supported files, cleans them,
    and returns a flat list of text chunks.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, docs_dir: str) -> List[Dict]:
        """
        Process all supported documents in docs_dir (recursively).

        Returns:
            List of chunk dicts: [{"text", "source", "chunk_id"}, ...]
        """
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: '{docs_dir}'")

        all_chunks: List[Dict] = []

        for file_path in sorted(docs_path.rglob("*")):
            suffix = file_path.suffix.lower()
            if suffix not in LOADERS:
                continue

            rel_name = str(file_path.relative_to(docs_path))
            print(f"  [Ingest] Loading '{rel_name}' ...")

            try:
                raw_text = LOADERS[suffix](str(file_path))
                clean = _clean_text(raw_text)
                if not clean:
                    print(f"  [Ingest] Warning: empty content in '{rel_name}', skipping.")
                    continue
                chunks = _chunk_text(
                    clean,
                    source=rel_name,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                all_chunks.extend(chunks)
                print(f"  [Ingest] → {len(chunks)} chunks from '{rel_name}'")
            except Exception as exc:
                print(f"  [Ingest] Error loading '{rel_name}': {exc}")

        return all_chunks
