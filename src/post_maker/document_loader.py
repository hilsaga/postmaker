"""Scan and parse supported documents under ``inputs/``."""

from __future__ import annotations

import logging
from pathlib import Path

from post_maker.config import (
    SUPPORTED_DOCX_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
)
from post_maker.models import ParsedDocument, TextChunk

logger = logging.getLogger(__name__)


def discover_files(inputs_dir: Path) -> list[Path]:
    """Recursively discover supported files under ``inputs_dir``.

    Args:
        inputs_dir: Directory to scan.

    Returns:
        Sorted list of matching file paths. Unsupported files are skipped
        with a log message.
    """
    if not inputs_dir.exists():
        logger.error("Inputs directory missing: %s", inputs_dir)
        return []
    if not inputs_dir.is_dir():
        logger.error("Inputs path is not a directory: %s", inputs_dir)
        return []

    found: list[Path] = []
    for path in sorted(inputs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            found.append(path)
        else:
            logger.info("Skipping unsupported file format: %s", path.name)
    logger.info("Discovered %d supported file(s) in %s", len(found), inputs_dir)
    return found


def parse_text_file(path: Path) -> ParsedDocument:
    """Parse a plain-text or Markdown file into line-based chunks.

    Args:
        path: Path to a ``.txt`` or ``.md`` file.

    Returns:
        :class:`ParsedDocument` with per-line chunks.
    """
    doc = ParsedDocument(
        path=path,
        document_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
    )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        doc.error = f"Failed to read text file: {exc}"
        logger.error("%s (%s)", doc.error, path)
        return doc

    for idx, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            doc.chunks.append(TextChunk(text=line, line_number=idx))
    if not doc.chunks and text.strip():
        doc.chunks.append(TextChunk(text=text.strip(), line_number=1))
    return doc


def parse_pdf_file(path: Path, *, max_pages: int | None = None) -> ParsedDocument:
    """Extract text from a PDF using pdfplumber, with pypdf fallback.

    When the PDF has no extractable text layer (scanned/image PDF), page images
    are rendered with ``pypdfium2`` so a later vision/OCR step can read them.

    Args:
        path: Path to a ``.pdf`` file.
        max_pages: Optional cap on pages to render for scanned PDFs.

    Returns:
        :class:`ParsedDocument` with page-numbered chunks and/or page images.
    """
    doc = ParsedDocument(path=path, document_name=path.name, file_type="pdf")
    page_count: int | None = None

    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None  # type: ignore[assignment]

    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        doc.chunks.append(TextChunk(text=text, page_number=page_num))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber failed for %s (%s); trying pypdf.", path.name, exc)

    if not doc.chunks:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    doc.chunks.append(TextChunk(text=text, page_number=page_num))
        except Exception as exc:  # noqa: BLE001
            if page_count is None:
                doc.error = f"Failed to parse PDF: {exc}"
                logger.error("%s (%s)", doc.error, path)
                return doc
            logger.warning("pypdf text extract failed for %s: %s", path.name, exc)

    doc.metadata["page_count"] = page_count
    if doc.chunks:
        doc.metadata["text_source"] = "embedded"
        # Page images are rendered later after the user selects which pages to read.
        return doc

    # Scanned / image-only PDF: defer rendering until the user picks page numbers.
    logger.warning(
        "No embedded text in %s (likely a scanned PDF). "
        "Page images will be rendered after you choose which pages to read.",
        path.name,
    )
    doc.metadata["text_source"] = "scanned"
    doc.metadata["needs_vision"] = True
    return doc


def render_pdf_pages(
    path: Path,
    *,
    pages: list[int] | None = None,
    max_pages: int | None = None,
    scale: float = 1.5,
    cache_dir: Path | None = None,
) -> list[Path]:
    """Render PDF pages to JPEG files using pypdfium2.

    Args:
        path: PDF file path.
        pages: Specific 1-based page numbers to render. Takes priority when set.
        max_pages: Optional maximum number of pages to render from the start
            when ``pages`` is not provided.
        scale: Render scale factor (1.5 ≈ 108 dpi * 1.5).
        cache_dir: Directory for rendered JPEGs. Defaults to
            ``.cache/pdf_pages/<pdf_stem>/`` under the project root.

    Returns:
        Paths to rendered JPEG page images.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        logger.error("pypdfium2 is required to render scanned PDFs: %s", exc)
        return []

    from post_maker.config import PROJECT_ROOT

    out_dir = cache_dir or (PROJECT_ROOT / ".cache" / "pdf_pages" / path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to open PDF for rendering (%s): %s", path.name, exc)
        return []

    total = len(pdf)
    if pages:
        page_list = sorted({p for p in pages if 1 <= p <= total})
    elif max_pages is not None and max_pages > 0:
        page_list = list(range(1, min(total, max_pages) + 1))
    else:
        page_list = list(range(1, total + 1))

    rendered: list[Path] = []
    for page_num in page_list:
        out_path = out_dir / f"page_{page_num:03d}.jpg"
        try:
            if out_path.exists() and out_path.stat().st_size > 0:
                rendered.append(out_path)
                continue
            page = pdf[page_num - 1]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            image.save(out_path, format="JPEG", quality=85, optimize=True)
            rendered.append(out_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to render %s page %d: %s", path.name, page_num, exc)

    return rendered


def apply_pdf_page_selection(document: ParsedDocument, pages: list[int]) -> ParsedDocument:
    """Keep only selected PDF pages (text + rendered images).

    Args:
        document: Parsed PDF document.
        pages: 1-based page numbers chosen by the user.

    Returns:
        The same document instance, mutated in place.
    """
    if document.file_type != "pdf":
        return document

    page_count = int(document.metadata.get("page_count") or 0)
    valid = sorted({p for p in pages if page_count == 0 or 1 <= p <= page_count})
    if not valid:
        logger.warning("No valid pages selected for %s; defaulting to page 1.", document.document_name)
        valid = [1]

    document.metadata["selected_pages"] = valid

    if document.chunks:
        document.chunks = [
            chunk
            for chunk in document.chunks
            if chunk.page_number is None or chunk.page_number in valid
        ]

    rendered = render_pdf_pages(document.path, pages=valid)
    document.image_paths = rendered
    document.metadata["rendered_pages"] = len(rendered)

    # Always vision-read selected PDF page images. Catalog product names are often
    # only in artwork even when sparse embedded text exists (e.g. blind-box sheets).
    if rendered:
        document.metadata["needs_vision"] = True
        if not document.chunks:
            document.metadata["text_source"] = document.metadata.get("text_source") or "scanned"
        logger.info(
            "Marked %s for vision product extraction (%d page image(s)).",
            document.document_name,
            len(rendered),
        )
    elif not document.chunks:
        document.metadata["needs_vision"] = True
        document.metadata["text_source"] = document.metadata.get("text_source") or "scanned"
    else:
        document.metadata["needs_vision"] = False

    logger.info(
        "Prepared %s pages %s (%d text chunk(s), %d image(s)).",
        document.document_name,
        valid,
        len(document.chunks),
        len(document.image_paths),
    )
    return document


def parse_number_list(raw: str, *, upper_bound: int | None = None) -> list[int]:
    """Parse a comma-separated list of numbers and ranges like ``1,3,5-7``.

    Args:
        raw: User-entered selection string.
        upper_bound: Optional upper bound for validation (inclusive).

    Returns:
        Sorted unique 1-based numbers.
    """
    numbers: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, _, right = token.partition("-")
            try:
                start = int(left.strip())
                end = int(right.strip())
            except ValueError:
                logger.warning("Ignoring invalid range: %s", token)
                continue
            if start > end:
                start, end = end, start
            for number in range(start, end + 1):
                if number < 1:
                    continue
                if upper_bound and number > upper_bound:
                    logger.warning(
                        "Number %d is outside 1..%d; skipping.", number, upper_bound
                    )
                    continue
                numbers.add(number)
            continue
        try:
            number = int(token)
        except ValueError:
            logger.warning("Ignoring invalid number: %s", token)
            continue
        if number < 1:
            continue
        if upper_bound and number > upper_bound:
            logger.warning(
                "Number %d is outside 1..%d; skipping.", number, upper_bound
            )
            continue
        numbers.add(number)
    return sorted(numbers)


def parse_page_list(raw: str, *, page_count: int | None = None) -> list[int]:
    """Parse a comma-separated page list like ``1,3,5-7``.

    Args:
        raw: User-entered page string.
        page_count: Optional upper bound for validation.

    Returns:
        Sorted unique 1-based page numbers.
    """
    return parse_number_list(raw, upper_bound=page_count)


def prompt_for_pdf_pages(
    documents: list[ParsedDocument],
    *,
    preselected: list[int] | None = None,
) -> dict[str, list[int]]:
    """Ask which PDF page(s) to read (comma-separated).

    Args:
        documents: Parsed documents; only PDFs are prompted.
        preselected: Optional CLI-provided page numbers applied to every PDF.

    Returns:
        Mapping of document name → selected page numbers.
    """
    pdf_docs = [doc for doc in documents if doc.file_type == "pdf" and not doc.error]
    selections: dict[str, list[int]] = {}
    if not pdf_docs:
        return selections

    for doc in pdf_docs:
        page_count = int(doc.metadata.get("page_count") or 0)
        if preselected:
            pages = [
                p
                for p in preselected
                if page_count == 0 or 1 <= p <= page_count
            ] or [1]
            selections[doc.document_name] = pages
            print(
                f"Using pages {pages} for PDF '{doc.document_name}' "
                f"({page_count or '?'} pages total)."
            )
            continue

        print(
            f"\nPDF found: {doc.document_name} "
            f"({page_count or '?'} pages)."
        )
        print("Which page(s) should be read? Separate by commas (e.g. 1,2,5 or 1-3,8).")
        raw = input("Pages [1]: ").strip()
        if not raw:
            pages = [1]
        else:
            pages = parse_page_list(raw, page_count=page_count or None)
            if not pages:
                print("  No valid pages parsed; defaulting to page 1.")
                pages = [1]
        selections[doc.document_name] = pages
        logger.info("Selected pages for %s: %s", doc.document_name, pages)

    return selections


def parse_docx_file(path: Path) -> ParsedDocument:
    """Extract paragraphs from a Word document.

    Args:
        path: Path to a ``.docx`` file.

    Returns:
        :class:`ParsedDocument` with section/paragraph chunks.
    """
    doc = ParsedDocument(path=path, document_name=path.name, file_type="docx")
    try:
        from docx import Document
    except ImportError as exc:
        doc.error = f"python-docx is not installed: {exc}"
        logger.error(doc.error)
        return doc

    try:
        document = Document(str(path))
        for idx, paragraph in enumerate(document.paragraphs, start=1):
            text = paragraph.text.strip()
            if text:
                style_name = getattr(paragraph.style, "name", None) or f"Paragraph {idx}"
                doc.chunks.append(
                    TextChunk(text=text, section=style_name, line_number=idx)
                )
    except Exception as exc:  # noqa: BLE001
        doc.error = f"Failed to parse DOCX: {exc}"
        logger.error("%s (%s)", doc.error, path)
    return doc


def parse_image_file(path: Path, *, use_ocr: bool = False) -> ParsedDocument:
    """Register an image file and optionally OCR its text with pytesseract.

    Args:
        path: Path to an image file.
        use_ocr: When True, attempt OCR if ``pytesseract`` is available.

    Returns:
        :class:`ParsedDocument` with the image path attached.
    """
    doc = ParsedDocument(
        path=path,
        document_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        image_paths=[path],
    )

    # Validate the image can be opened.
    try:
        from PIL import Image

        with Image.open(path) as img:
            doc.metadata["size"] = img.size
            doc.metadata["mode"] = img.mode
    except Exception as exc:  # noqa: BLE001
        doc.error = f"Failed to open image: {exc}"
        logger.error("%s (%s)", doc.error, path)
        return doc

    if use_ocr:
        try:
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(path)).strip()
            if text:
                doc.chunks.append(TextChunk(text=text, section="OCR"))
        except ImportError:
            logger.info(
                "OCR requested for %s but pytesseract is not installed; skipping OCR.",
                path.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed for %s: %s", path.name, exc)

    return doc


def parse_file(
    path: Path,
    *,
    use_ocr: bool = False,
    max_pages: int | None = None,
) -> ParsedDocument:
    """Dispatch parsing based on file extension.

    Args:
        path: File to parse.
        use_ocr: Enable OCR for image files when available.
        max_pages: Max pages to render for scanned PDFs.

    Returns:
        Parsed document (may include an ``error`` message).
    """
    if not path.exists():
        logger.error("File does not exist: %s", path)
        return ParsedDocument(
            path=path,
            document_name=path.name,
            file_type="unknown",
            error="File does not exist",
        )

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return parse_text_file(path)
    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return parse_pdf_file(path, max_pages=max_pages)
    if suffix in SUPPORTED_DOCX_EXTENSIONS:
        return parse_docx_file(path)
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return parse_image_file(path, use_ocr=use_ocr)

    logger.warning("Unsupported extension for %s", path.name)
    return ParsedDocument(
        path=path,
        document_name=path.name,
        file_type=suffix.lstrip(".") or "unknown",
        error=f"Unsupported file format: {suffix}",
    )


def load_documents(
    inputs_dir: Path,
    *,
    use_ocr: bool = False,
    max_pages: int | None = None,
) -> list[ParsedDocument]:
    """Discover and parse all supported documents under ``inputs_dir``.

    Args:
        inputs_dir: Directory to scan.
        use_ocr: Enable OCR for image files when available.
        max_pages: Max pages to render for scanned PDFs.

    Returns:
        List of parsed documents (including any that failed with errors).
    """
    files = discover_files(inputs_dir)
    if not files:
        logger.warning("No supported documents found in %s", inputs_dir)
        return []

    documents: list[ParsedDocument] = []
    for path in files:
        rel = path.relative_to(inputs_dir) if path.is_relative_to(inputs_dir) else path
        logger.info("Parsing %s", rel)
        documents.append(parse_file(path, use_ocr=use_ocr, max_pages=max_pages))
    return documents


def print_document_inventory(documents: list[ParsedDocument]) -> None:
    """Print a console inventory of what was read from inputs.

    Args:
        documents: Parsed documents from the loader.
    """
    print("\n=== Documents read from inputs ===")
    if not documents:
        print("  (none)")
        return

    for doc in documents:
        pages = doc.metadata.get("page_count")
        source = doc.metadata.get("text_source")
        chunk_count = len(doc.chunks)
        image_count = len(doc.image_paths)
        status = "ERROR" if doc.error else "OK"
        detail_parts = [f"type={doc.file_type}", f"text_chunks={chunk_count}", f"images={image_count}"]
        if pages is not None:
            detail_parts.append(f"pages={pages}")
        if source:
            detail_parts.append(f"source={source}")
        if doc.metadata.get("needs_vision"):
            detail_parts.append("needs_vision=yes")
        print(f"  [{status}] {doc.document_name} ({', '.join(detail_parts)})")
        if doc.error:
            print(f"         error: {doc.error}")
        elif chunk_count:
            preview = " ".join(doc.full_text.split())[:160]
            if preview:
                print(f"         preview: {preview}…")
        elif image_count:
            print(f"         page/image files: {image_count} (awaiting vision extraction)")

