"""Tool: extract plain text from an uploaded resume (PDF or TXT)."""
from __future__ import annotations

import io
import re


def extract_resume_text(file_bytes: bytes, filename: str, max_chars: int = 12000) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(
            "No text could be read from this file. If it is a scanned image PDF, "
            "type your skills in the box instead."
        )
    return text[:max_chars]
