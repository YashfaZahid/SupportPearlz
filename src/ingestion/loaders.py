"""File-type dispatch loaders with metadata contract and per-file error handling."""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from src.ingestion.normalize import normalize_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".csv", ".docx"}

DOC_TYPE_BY_HINT: list[tuple[str, str]] = [
    ("manual", "manual"),
    ("warranty", "policy"),
    ("refund", "policy"),
    ("return", "policy"),
    ("shipping", "policy"),
    ("privacy", "policy"),
    ("install", "guide"),
    ("troubleshoot", "guide"),
    ("faq", "faq"),
    ("pricing", "pricing"),
    ("service", "policy"),
    ("handbook", "guide"),
    ("changelog", "guide"),
    ("firmware", "guide"),
]

PRODUCT_LINE_HINTS = [
    "AquaPearl 500 Pro",
    "AquaPearl 300",
    "AquaPearl 200",
    "AeroPearl Mini",
    "AeroPearl Classic",
]


@dataclass
class FileIngestionResult:
    path: Path
    status: str  # loaded | skipped | failed
    reason: str = ""
    detected_type: str = ""
    char_count: int = 0
    page_or_section_count: int = 0
    metadata_keys: list[str] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)


@dataclass
class IngestionReport:
    files_found: int = 0
    loaded: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[FileIngestionResult] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Ingestion report: found={self.files_found} loaded={self.loaded} "
            f"skipped={self.skipped} failed={self.failed}",
        ]
        for r in self.results:
            lines.append(
                f"  [{r.status}] {r.path.name} type={r.detected_type or '-'} "
                f"chars={r.char_count} units={r.page_or_section_count} "
                f"meta={r.metadata_keys} {r.reason}".rstrip()
            )
        return lines


def infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    for hint, doc_type in DOC_TYPE_BY_HINT:
        if hint in lower:
            return doc_type
    return "guide"


def infer_product_line(filename: str, text_sample: str = "") -> str:
    blob = f"{filename}\n{text_sample[:2000]}"
    for name in PRODUCT_LINE_HINTS:
        if name.lower().replace(" ", "") in blob.lower().replace(" ", "") or name.lower() in blob.lower():
            return name
    return "general"


def content_fingerprint(text: str) -> str:
    normalised = normalize_text(text).encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()


def infer_access_level(filename: str) -> str:
    """C6: internal handbook vs public customer docs."""
    lower = filename.lower()
    if "handbook" in lower or "internal" in lower:
        return "internal"
    return "public"


def _base_metadata(
    path: Path,
    *,
    doc_type: str,
    product_line: str,
    version: str = "2026.1",
    location: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "source": path.name,
        "source_path": str(path.resolve()),
        "doc_type": doc_type,
        "product_line": product_line,
        "version": version,
        "last_updated": version,
        "location": location,
        "access_level": infer_access_level(path.name),
    }
    if extra:
        meta.update(extra)
    return meta


def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    doc_type = infer_doc_type(path.name)
    out: list[Document] = []
    for page in pages:
        text = normalize_text(page.page_content)
        if not text:
            continue
        page_num = page.metadata.get("page", page.metadata.get("page_number", 0))
        # PyPDFLoader often uses 0-based page index
        location = f"page:{int(page_num) + 1}" if page_num is not None else "page:unknown"
        product_line = infer_product_line(path.name, text)
        meta = _base_metadata(
            path,
            doc_type=doc_type,
            product_line=product_line,
            location=location,
            extra={"page": int(page_num) + 1 if page_num is not None else None},
        )
        # Preserve loader metadata keys we do not overwrite intentionally
        for k, v in page.metadata.items():
            if k not in meta:
                meta[k] = v
        out.append(Document(page_content=text, metadata=meta))
    return out


def _load_text_like(path: Path) -> list[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    doc_type = infer_doc_type(path.name)
    out: list[Document] = []
    for doc in docs:
        text = normalize_text(doc.page_content)
        if not text:
            continue
        # Prefer first markdown heading as location
        location = "document"
        for line in text.splitlines():
            if line.startswith("#"):
                location = line.lstrip("#").strip()[:120] or "document"
                break
        product_line = infer_product_line(path.name, text)
        meta = _base_metadata(
            path,
            doc_type=doc_type,
            product_line=product_line,
            location=f"section:{location}",
        )
        out.append(Document(page_content=text, metadata=meta))
    return out


def _load_docx(path: Path) -> list[Document]:
    # Docx2txtLoader requires optional docx2txt; fall back to python-docx if missing.
    try:
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        texts = [normalize_text(d.page_content) for d in docs]
    except Exception:
        from docx import Document as DocxFile

        document = DocxFile(str(path))
        texts = [normalize_text("\n".join(p.text for p in document.paragraphs if p.text))]

    doc_type = infer_doc_type(path.name)
    out: list[Document] = []
    for text in texts:
        if not text:
            continue
        location = "document"
        for line in text.splitlines():
            if line.strip():
                location = line.strip()[:120]
                break
        product_line = infer_product_line(path.name, text)
        meta = _base_metadata(
            path,
            doc_type=doc_type,
            product_line=product_line,
            location=f"section:{location}",
        )
        out.append(Document(page_content=text, metadata=meta))
    return out


def _load_csv(path: Path) -> list[Document]:
    """Row-wise documents so SKU and price stay together."""
    doc_type = infer_doc_type(path.name)
    out: list[Document] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return out
        for i, row in enumerate(reader, start=1):
            sku = (row.get("sku") or "").strip()
            name = (row.get("product_name") or "").strip()
            desc = (row.get("description") or "").strip()
            list_price = (row.get("list_price_pkr") or "").strip()
            promo = (row.get("promo_price_pkr") or "").strip()
            availability = (row.get("availability") or "").strip()
            product_line = (row.get("product_line") or infer_product_line(path.name)).strip()
            sentence = (
                f"SKU {sku}: {name}. {desc}. "
                f"List price PKR {list_price}. "
                f"Current promotional price PKR {promo}. "
                f"Availability: {availability}. "
                f"Product line: {product_line}."
            )
            text = normalize_text(sentence)
            if not text:
                continue
            meta = _base_metadata(
                path,
                doc_type=doc_type or "pricing",
                product_line=product_line or "general",
                location=f"row:{i}|sku:{sku or 'unknown'}",
                extra={"sku": sku, "row": i},
            )
            out.append(Document(page_content=text, metadata=meta))
    return out


def load_file(path: Path) -> list[Document]:
    """Dispatch a single file to the appropriate loader."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".md", ".txt"}:
        return _load_text_like(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported extension: {ext}")


def iter_knowledge_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("_") or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield path


def load_knowledge_base(root: Path) -> tuple[list[Document], IngestionReport]:
    """Walk ``root``, load supported files, skip corrupt files, dedupe by content hash."""
    report = IngestionReport()
    documents: list[Document] = []
    seen_hashes: set[str] = set()

    files = list(iter_knowledge_files(root))
    report.files_found = len(files)

    for path in files:
        try:
            loaded = load_file(path)
        except Exception as exc:  # per-file failure must not abort the run
            logger.warning("Failed to load %s: %s", path.name, exc)
            report.failed += 1
            report.results.append(
                FileIngestionResult(
                    path=path,
                    status="failed",
                    reason=str(exc),
                    detected_type=path.suffix.lower(),
                )
            )
            continue

        if not loaded:
            report.skipped += 1
            report.results.append(
                FileIngestionResult(
                    path=path,
                    status="skipped",
                    reason="empty after normalisation",
                    detected_type=path.suffix.lower(),
                )
            )
            continue

        kept: list[Document] = []
        for doc in loaded:
            if not doc.metadata.get("source") or not doc.metadata.get("doc_type"):
                raise RuntimeError(f"Metadata contract violated for {path}")
            fp = content_fingerprint(doc.page_content)
            if fp in seen_hashes:
                logger.info("Skipping duplicate content from %s", path.name)
                continue
            seen_hashes.add(fp)
            doc.metadata["content_hash"] = fp
            kept.append(doc)

        if not kept:
            report.skipped += 1
            report.results.append(
                FileIngestionResult(
                    path=path,
                    status="skipped",
                    reason="duplicate content",
                    detected_type=path.suffix.lower(),
                )
            )
            continue

        char_count = sum(len(d.page_content) for d in kept)
        meta_keys = sorted({k for d in kept for k in d.metadata})
        report.loaded += 1
        report.results.append(
            FileIngestionResult(
                path=path,
                status="loaded",
                detected_type=path.suffix.lower(),
                char_count=char_count,
                page_or_section_count=len(kept),
                metadata_keys=meta_keys,
                documents=kept,
            )
        )
        documents.extend(kept)

    return documents, report
