"""
PDF Extraction Module - Minor Project Version
Uses PyMuPDF (fitz) to extract text, document metadata, and structural sections.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

try:
    import pymupdf as fitz
except ImportError:
    import fitz


# Common academic banners to ignore when detecting title
HEADER_IGNORE_PATTERNS = [
    r"(?i)^arxiv:\d+\.\d+",
    r"(?i)^ieee\s+transactions",
    r"(?i)^proceedings\s+of",
    r"(?i)^journal\s+of",
    r"(?i)^springer",
    r"(?i)^elsevier",
    r"(?i)^acm\s+conference",
    r"(?i)^preprint",
    r"(?i)^volume\s+\d+",
    r"(?i)^doi:\s*10\.\d+"
]

# Patterns representing affiliation noise in author blocks
AFFILIATION_PATTERNS = [
    r"(?i)department\s+of",
    r"(?i)school\s+of",
    r"(?i)faculty\s+of",
    r"(?i)university",
    r"(?i)institute",
    r"(?i)laboratory",
    r"(?i)college",
    r"(?i)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
]


def clean_text(raw_text: str) -> str:
    """Removes non-printable characters, normalizes ligatures and whitespace."""
    if not raw_text:
        return ""
    # Normalize common LaTeX typographical ligatures & smart quotes
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " - ",
        "\u2013": " - "
    }
    text = raw_text
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Fix broken hyphenations at line endings (e.g. 'classifi-\ncation' -> 'classification')
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Normalize excessive newlines and spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_title(full_text: str, page1_text: str, meta: Dict[str, Any], filename: str) -> str:
    """Extracts paper title from metadata or Page 1 text."""
    # 1. Embedded PDF metadata
    pdf_title = (meta.get("title") or "").strip()
    if pdf_title and len(pdf_title) > 6 and not any(pdf_title.lower().startswith(x) for x in ["untitled", "microsoft", "latex", "document"]):
        return pdf_title

    # 2. Page 1 text heuristics
    if page1_text:
        lines = [line.strip() for line in page1_text.split("\n") if line.strip()]
        candidate_lines = []
        for line in lines[:8]:
            if any(re.search(pat, line) for pat in HEADER_IGNORE_PATTERNS):
                continue
            if re.search(r"(?i)\b(abstract|introduction|university|department|email)\b", line):
                break
            if len(candidate_lines) >= 1 and ("," in line or " and " in line.lower() or "@" in line):
                break
            candidate_lines.append(line)
            if len(candidate_lines) >= 2:
                break
        if candidate_lines:
            candidate = " ".join(candidate_lines).strip()
            if 10 <= len(candidate) <= 250:
                return candidate

    # 3. Filename fallback
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return stem.title() or "Untitled Research Paper"


def extract_authors(page1_text: str, meta: Dict[str, Any]) -> str:
    """Extracts author names from metadata or Page 1 block."""
    meta_author = (meta.get("author") or "").strip()
    if meta_author and len(meta_author) > 2 and not any(meta_author.lower().startswith(x) for x in ["user", "admin", "author"]):
        return meta_author

    if not page1_text:
        return "Unknown Authors"

    # Search lines before the Abstract
    abstract_idx = re.search(r"(?i)\babstract\b", page1_text)
    header_chunk = page1_text[:abstract_idx.start()] if abstract_idx else page1_text[:1200]
    lines = [line.strip() for line in header_chunk.split("\n") if line.strip()]

    for line in lines[1:7]:
        if any(re.search(pat, line) for pat in HEADER_IGNORE_PATTERNS):
            continue
        if any(re.search(pat, line) for pat in AFFILIATION_PATTERNS):
            continue
        if ("," in line or " and " in line.lower()) and len(line) < 180:
            cleaned = re.sub(r"[\d\*\†\‡\§]+", "", line).strip()
            return cleaned

    return "Authors not identified"


def extract_publication_year(full_text: str, meta: Dict[str, Any]) -> int:
    """Detects 4-digit publication year from metadata or text."""
    current_year = datetime.now(timezone.utc).year

    # 1. Check embedded creation date (e.g. 'D:20230510')
    cdate = str(meta.get("creationDate") or meta.get("creation_date") or "")
    m = re.search(r"20\d{2}|19\d{2}", cdate)
    if m:
        y = int(m.group(0))
        if 1980 <= y <= current_year + 1:
            return y

    # 2. Check first 2000 characters for arXiv or copyright
    sample = full_text[:2000]
    arxiv_match = re.search(r"(?i)arxiv:(\d{2})\d{2}\.", sample)
    if arxiv_match:
        yy = int(arxiv_match.group(1))
        return 2000 + yy if yy <= (current_year % 100) + 1 else 1900 + yy

    year_match = re.search(r"(?i)(?:copyright|published|©|\bproceedings\b|\bconf\b)[^\d\n]*\b(19\d{2}|20\d{2})\b", sample)
    if year_match:
        y = int(year_match.group(1))
        if 1980 <= y <= current_year + 1:
            return y

    return current_year


def extract_abstract(full_text: str) -> str:
    """Extracts original abstract text using regex boundary matching."""
    if not full_text:
        return ""
    match = re.search(
        r"(?i)\babstract\b[:\s\-—]*(.+?)(?=\b(?:\d+\.?\s*)?introduction\b|\n\n\s*[1-9]\b|$)",
        full_text,
        re.DOTALL
    )
    if match:
        abs_text = match.group(1).strip()
        abs_text = re.sub(r"\s+", " ", abs_text)
        if 40 <= len(abs_text) <= 3500:
            return abs_text

    # Fallback to first paragraph
    paras = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 80]
    if paras and "abstract" in paras[0].lower()[:30]:
        return re.sub(r"\s+", " ", paras[0])

    return ""


def extract_sections(full_text: str) -> List[Dict[str, Any]]:
    """Segments the document into standard academic sections."""
    section_names = [
        "Abstract", "Introduction", "Related Work", "Methodology",
        "Results", "Discussion", "Conclusion", "References"
    ]
    pattern = r"(?im)^(?:\d+\.?\s*)?(" + "|".join(section_names) + r")\b"
    matches = list(re.finditer(pattern, full_text))

    if not matches:
        return [{"section_name": "Full Document", "content": full_text[:3000]}]

    sections = []
    for i, m in enumerate(matches):
        sec_name = m.group(1).title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        if content:
            sections.append({
                "section_name": sec_name,
                "section_order": i + 1,
                "content": content
            })

    return sections


def extract_pdf_data(file_path: Path) -> Dict[str, Any]:
    """
    Main entry point: opens PDF, extracts raw text, page metadata,
    and returns a structured dictionary.
    """
    doc = fitz.open(str(file_path))
    total_pages = len(doc)
    pages_text = []
    total_words = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        cleaned = clean_text(page.get_text("text") or "")
        words = len(cleaned.split()) if cleaned else 0
        total_words += words
        pages_text.append(cleaned)

    full_text = "\n\n".join(t for t in pages_text if t)
    page1_text = pages_text[0] if pages_text else ""
    meta = doc.metadata or {}

    title = extract_title(full_text, page1_text, meta, file_path.name)
    authors = extract_authors(page1_text, meta)
    pub_year = extract_publication_year(full_text, meta)
    abstract = extract_abstract(full_text)
    sections = extract_sections(full_text)

    doc.close()

    return {
        "title": title,
        "authors": authors,
        "publication_year": pub_year,
        "total_pages": total_pages,
        "total_words": total_words,
        "abstract": abstract,
        "sections": sections,
        "full_text": full_text
    }
