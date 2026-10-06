"""Information skill: read inputs, detect products, and search documents."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from post_maker.config import SUPPORTED_IMAGE_EXTENSIONS, Settings, load_settings
from post_maker.models import (
    DetectedCandidate,
    ItemMatch,
    ItemSummary,
    ParsedDocument,
    TextChunk,
    TokenUsageTracker,
)
from post_maker.vertex_client import generate_text

logger = logging.getLogger(__name__)

SNIPPET_RADIUS = 120


def _is_index_token(token: str) -> bool:
    """Return True when *token* is a single index or range like ``1`` or ``1-7``."""
    token = token.strip()
    if token.isdigit():
        return True
    if "-" in token:
        left, _, right = token.partition("-")
        return left.strip().isdigit() and right.strip().isdigit()
    return False


def _resolve_item_selection(
    raw: str,
    candidates: list[DetectedCandidate],
) -> list[str]:
    """Resolve numbers, ranges, names, or ``all`` to candidate product names."""
    if raw.strip().lower() in {"all", "*"}:
        return [candidate.name for candidate in candidates]

    selected: list[str] = []
    seen: set[str] = set()
    max_index = len(candidates)
    ignored: list[int] = []

    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if _is_index_token(token):
            if "-" in token:
                left, _, right = token.partition("-")
                start, end = int(left.strip()), int(right.strip())
                if start > end:
                    start, end = end, start
                indices = range(start, end + 1)
            else:
                indices = [int(token)]
            for index in indices:
                if 1 <= index <= max_index:
                    name = candidates[index - 1].name
                    key = _normalize(name)
                    if key not in seen:
                        seen.add(key)
                        selected.append(name)
                else:
                    ignored.append(index)
            continue

        key = _normalize(token)
        if key not in seen:
            seen.add(key)
            selected.append(token)

    if ignored:
        print(
            f"  *** Selection warning: ignored out-of-range number(s) {ignored} "
            f"(only {max_index} product(s) are listed above). ***"
        )
        print(
            "  If you expected more products (e.g. 1-7,9), the document listing "
            "did not detect that many individual items."
        )
    return selected

# Patterns for structured catalogs / markdown product lists.
_PRODUCT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^\s*(?:###?\s*)?product\s*[:\-–]\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*(?:item|sku|model)\s*[:\-–]\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*[-*•]\s+\*\*(.+?)\*\*"),
)
_CATEGORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^\s*(?:##\s*)?category\s*[:\-–]\s*(.+?)\s*$"),
)

_VISION_EXTRACT_PROMPT = """\
Read this catalog/document page image carefully.

1) Extract a concise plain-text transcription of the useful product copy.
2) List EVERY distinct physical product / figure / character / SKU that is
   CLEARLY VISIBLE or EXPLICITLY NAMED on the page.

Use exactly this format:

TRANSCRIPT:
- <short bullet lines of readable text>

PRODUCTS:
- <specific product/figure name — include accessory or pose if that distinguishes it>
- <specific product/figure name>

CATEGORIES:
- <category / series name>

CRITICAL RULES (do not violate):
- List EACH individual figure/SKU as its own PRODUCTS bullet. Do NOT collapse a
  whole series into one PRODUCTS line.
- Prefer specific names like "Mr. White Cloud with umbrella" over a series title.
- Series titles belong under CATEGORIES, not PRODUCTS (unless no individuals are visible).
- NEVER invent, guess, imply, or "complete" products. If a SECRET label exists but
  the secret figure is not shown/named, do NOT add a secret product.
- Do NOT write phrases like "implied by", "probably", "chase variant unknown".
- If nothing is found for a section, write "none" under it.
"""


_AI_PRODUCT_LIST_PROMPT = """\
From the attached catalog page image(s) and/or text, list EVERY distinct
physical product / figure / character / SKU that is CLEARLY VISIBLE or
EXPLICITLY NAMED.

Return ONLY a bullet list of specific product names (one per line), no preamble.

CRITICAL:
- One bullet per individual figure/SKU (e.g. "Mr. White Cloud with balloons").
- Do NOT collapse a series into a single bullet.
- NEVER invent or imply products (no "secret implied by label", no guessing).
- If a series title is all you can see with no individual members, list nothing
  except what is explicitly named.
"""


def _is_invented_or_weak_name(name: str) -> bool:
    """Reject AI names that invent/imply products not explicitly in the document."""
    lowered = name.lower()
    banned_fragments = (
        "implied",
        "probably",
        "maybe",
        "unknown",
        "not shown",
        "not pictured",
        "guess",
        "likely",
        "inferred",
        "suggested by",
        "chase variant",
        "secret variant (implied",
    )
    return any(fragment in lowered for fragment in banned_fragments)

def _normalize(value: str) -> str:
    """Normalize whitespace and case for matching."""
    return re.sub(r"\s+", " ", value).strip().lower()


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug from a product/item name."""
    slug = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "_", slug).strip("_").lower()
    return slug or "item"


def _snippet(text: str, start: int, end: int) -> str:
    """Build a short snippet around a match span."""
    left = max(0, start - SNIPPET_RADIUS)
    right = min(len(text), end + SNIPPET_RADIUS)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return f"{prefix}{text[left:right].strip()}{suffix}"


def _clean_name(raw: str) -> str:
    """Strip list markers / markdown noise from a detected name."""
    name = raw.strip().strip("-*• ").strip()
    name = re.sub(r"^\[\s*|\s*\]$", "", name)
    name = name.strip("`'\" ")
    return re.sub(r"\s+", " ", name).strip()


def _parse_vision_sections(text: str) -> tuple[str, list[str], list[str]]:
    """Parse TRANSCRIPT / PRODUCTS / CATEGORIES sections from a vision response."""
    transcript_lines: list[str] = []
    products: list[str] = []
    categories: list[str] = []
    section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")
        if upper in {"TRANSCRIPT", "PRODUCTS", "CATEGORIES"}:
            section = upper.lower()
            continue
        if not stripped or stripped.lower() == "none":
            continue
        cleaned = _clean_name(re.sub(r"^[-*•]\s*", "", stripped))
        if not cleaned:
            continue
        if section == "transcript":
            transcript_lines.append(cleaned)
        elif section == "products":
            products.append(cleaned)
        elif section == "categories":
            categories.append(cleaned)

    return "\n".join(transcript_lines), products, categories


class InformationSkill:
    """Read parsed inputs, detect products/items, and locate selections."""

    def __init__(
        self,
        documents: list[ParsedDocument],
        *,
        settings: Settings | None = None,
        tracker: TokenUsageTracker | None = None,
    ) -> None:
        """
        Args:
            documents: Previously parsed documents to search.
            settings: Optional settings for Vertex vision enrichment.
            tracker: Optional shared token-usage tracker.
        """
        self.documents = documents
        self.settings = settings or load_settings()
        self.tracker = tracker if tracker is not None else TokenUsageTracker()

    def enrich_scanned_documents(self, *, max_images_per_doc: int | None = None) -> None:
        """Use Vertex vision to OCR/extract text from scanned PDFs and images.

        Extracted text is written back into each document's ``chunks`` so later
        search/detection operates on real content from ``inputs/``.

        Args:
            max_images_per_doc: Optional cap on images/pages processed per document.
        """
        for doc in self.documents:
            if doc.error:
                continue
            needs_vision = bool(doc.metadata.get("needs_vision")) or (
                not doc.chunks and bool(doc.image_paths)
            ) or (
                # PDF page renders should always be vision-read for product discovery.
                doc.file_type == "pdf" and bool(doc.image_paths)
            )
            if not needs_vision:
                continue

            # Skip pages already vision-enriched in this run.
            already = {
                chunk.page_number
                for chunk in doc.chunks
                if chunk.section == "vision" and chunk.page_number is not None
            }

            images = list(doc.image_paths)
            if max_images_per_doc and max_images_per_doc > 0:
                images = images[:max_images_per_doc]
            if not images:
                logger.warning("Document %s needs vision but has no images.", doc.document_name)
                continue

            print(
                f"\nReading products from {doc.document_name} "
                f"via Vertex vision ({len(images)} page/image(s))…"
            )
            had_embedded = any(chunk.section != "vision" for chunk in doc.chunks)
            for index, image_path in enumerate(images, start=1):
                page_num = self._page_num_from_image(image_path, fallback=index)
                if page_num in already:
                    continue
                step = f"vision:{_slugify(doc.document_name)}:page_{page_num}"
                result = generate_text(
                    _VISION_EXTRACT_PROMPT,
                    settings=self.settings,
                    step=step,
                    tracker=self.tracker,
                    image_paths=[image_path],
                )
                if not result.text:
                    logger.warning("Vision returned no text for %s page %s", doc.document_name, page_num)
                    continue

                transcript, products, categories = _parse_vision_sections(result.text)
                products = [
                    name for name in products if name and not _is_invented_or_weak_name(name)
                ]
                body_parts = [transcript] if transcript else [result.text]
                if products:
                    # One product per line so detection never loses individuals.
                    body_parts.append("Products:")
                    body_parts.extend(f"- {name}" for name in products)
                    body_parts.append("Products: " + "; ".join(products))
                if categories:
                    body_parts.append("Categories: " + "; ".join(categories))
                chunk_text = "\n".join(part for part in body_parts if part).strip()
                doc.chunks.append(
                    TextChunk(
                        text=chunk_text,
                        page_number=page_num,
                        section="vision",
                    )
                )
                existing = list(doc.metadata.get("vision_products") or [])
                for name in products:
                    if _normalize(name) not in {_normalize(x) for x in existing}:
                        existing.append(name)
                doc.metadata["vision_products"] = existing
                logger.info(
                    "Extracted vision text for %s page %s (%d chars, %d products).",
                    doc.document_name,
                    page_num,
                    len(chunk_text),
                    len(products),
                )
                if products:
                    print(f"  Page {page_num}: found {len(products)} product(s)")
                    for name in products[:20]:
                        print(f"    - {name}")
                    if len(products) > 20:
                        print(f"    … +{len(products) - 20} more")
                else:
                    print(
                        f"  Page {page_num}: no individual products listed "
                        "(refusing invented/implied names)."
                    )

            doc.metadata["needs_vision"] = False
            if had_embedded:
                doc.metadata["text_source"] = "embedded+vision"
            else:
                doc.metadata["text_source"] = "vertex_vision"
            if not doc.chunks:
                logger.warning("Vision enrichment produced no chunks for %s", doc.document_name)

    @staticmethod
    def _page_num_from_image(image_path: Path, *, fallback: int) -> int:
        """Infer a page number from names like ``page_003.jpg``."""
        match = re.search(r"page[_\-]?(\d+)", image_path.stem, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return fallback

    def detect_candidates(self) -> list[DetectedCandidate]:
        """Detect products/items/categories from already-read document text.

        Returns:
            Deduplicated list of :class:`DetectedCandidate` values discovered
            in the parsed ``inputs/`` content.
        """
        found: list[DetectedCandidate] = []
        seen: set[str] = set()

        # Prefer explicit vision product lists stored during enrichment.
        for doc in self.documents:
            if doc.error:
                continue
            for name in doc.metadata.get("vision_products") or []:
                cleaned = _clean_name(str(name))
                if _is_invented_or_weak_name(cleaned):
                    continue
                self._append_candidate(
                    found,
                    seen,
                    DetectedCandidate(
                        name=cleaned,
                        kind="product",
                        source_document=doc.document_name,
                        location="vision products",
                        snippet=cleaned,
                    ),
                )

        for doc in self.documents:
            if doc.error:
                continue

            # Structured text patterns.
            for chunk in doc.chunks:
                for pattern in _PRODUCT_PATTERNS:
                    for match in pattern.finditer(chunk.text):
                        name = _clean_name(match.group(1))
                        if _is_invented_or_weak_name(name):
                            continue
                        self._append_candidate(
                            found,
                            seen,
                            DetectedCandidate(
                                name=name,
                                kind="product",
                                source_document=doc.document_name,
                                location=chunk.location_label(),
                                snippet=_snippet(chunk.text, match.start(1), match.end(1)),
                            ),
                        )
                for pattern in _CATEGORY_PATTERNS:
                    for match in pattern.finditer(chunk.text):
                        name = _clean_name(match.group(1))
                        if _is_invented_or_weak_name(name):
                            continue
                        self._append_candidate(
                            found,
                            seen,
                            DetectedCandidate(
                                name=name,
                                kind="category",
                                source_document=doc.document_name,
                                location=chunk.location_label(),
                                snippet=_snippet(chunk.text, match.start(1), match.end(1)),
                            ),
                        )

                # Vision section: "Products: A; B; C" and/or bullet lists after "Products:"
                products_line = re.search(r"(?im)^products?\s*:\s*(.+)$", chunk.text)
                if products_line and products_line.group(1).strip():
                    for part in re.split(r"[;•|\n]", products_line.group(1)):
                        name = _clean_name(part)
                        if len(name) < 3 or _is_invented_or_weak_name(name):
                            continue
                        self._append_candidate(
                            found,
                            seen,
                            DetectedCandidate(
                                name=name,
                                kind="product",
                                source_document=doc.document_name,
                                location=chunk.location_label(),
                                snippet=name,
                            ),
                        )
                # Bullets under a Products header in the same chunk.
                if re.search(r"(?im)^products?\s*:\s*$", chunk.text) or re.search(
                    r"(?im)^products?\s*:", chunk.text
                ):
                    for line in chunk.text.splitlines():
                        bullet = re.match(r"^\s*[-*•]\s+(.+)$", line)
                        if not bullet:
                            continue
                        name = _clean_name(bullet.group(1))
                        if len(name) < 3 or _is_invented_or_weak_name(name):
                            continue
                        self._append_candidate(
                            found,
                            seen,
                            DetectedCandidate(
                                name=name,
                                kind="product",
                                source_document=doc.document_name,
                                location=chunk.location_label(),
                                snippet=name,
                            ),
                        )

            # Image filenames as weak candidates (product photos).
            for image in doc.image_paths:
                stem = image.stem
                if re.fullmatch(r"page[_\-]?\d+", stem, flags=re.IGNORECASE):
                    continue
                name = stem.replace("_", " ").replace("-", " ").strip()
                if len(name) < 3 or _is_invented_or_weak_name(name):
                    continue
                self._append_candidate(
                    found,
                    seen,
                    DetectedCandidate(
                        name=name.title(),
                        kind="item",
                        source_document=doc.document_name,
                        location="filename",
                        snippet=image.name,
                    ),
                )

        logger.info("Detected %d candidate item(s)/product(s) from inputs.", len(found))
        return found

    def extract_candidates_with_ai(self) -> list[DetectedCandidate]:
        """Ask Gemini to list products from already-loaded text/images.

        Used as a fallback when regex/vision-section parsing finds nothing.

        Returns:
            Deduplicated :class:`DetectedCandidate` list.
        """
        found: list[DetectedCandidate] = []
        seen: set[str] = set()

        for doc in self.documents:
            if doc.error:
                continue
            context = doc.full_text.strip()
            images = [path for path in doc.image_paths if path.exists()][:4]
            if not context and not images:
                continue

            prompt = (
                f"{_AI_PRODUCT_LIST_PROMPT}\n\n"
                f"Document: {doc.document_name}\n"
                f"Text context:\n{context[:4000] or '(no text)'}\n"
            )
            step = f"vision:{_slugify(doc.document_name)}:product_list"
            result = generate_text(
                prompt,
                settings=self.settings,
                step=step,
                tracker=self.tracker,
                image_paths=images or None,
            )
            if not result.text:
                continue

            for line in result.text.splitlines():
                name = _clean_name(re.sub(r"^[-*•\d\.\)\s]+", "", line.strip()))
                if len(name) < 2 or name.lower() in {"none", "n/a"}:
                    continue
                if _is_invented_or_weak_name(name):
                    print(f"  Skipping invented/implied product name: {name}")
                    continue
                self._append_candidate(
                    found,
                    seen,
                    DetectedCandidate(
                        name=name,
                        kind="product",
                        source_document=doc.document_name,
                        location="AI extraction",
                        snippet=name,
                    ),
                )

        logger.info("AI extraction found %d candidate product(s).", len(found))
        return found

    @staticmethod
    def _append_candidate(
        found: list[DetectedCandidate],
        seen: set[str],
        candidate: DetectedCandidate,
    ) -> None:
        """Append a candidate if its normalized name is new and non-empty."""
        if not candidate.name or len(candidate.name) < 2:
            return
        key = _normalize(candidate.name)
        if key in seen:
            return
        seen.add(key)
        found.append(candidate)

    def find_items(self, items: list[str]) -> list[ItemSummary]:
        """Search all documents for each requested item.

        Args:
            items: Item, product, or category names to locate.

        Returns:
            One :class:`ItemSummary` per requested item (even if not found).
        """
        cleaned = [item.strip() for item in items if item and item.strip()]
        if not cleaned:
            logger.warning("No items provided for search.")
            return []

        summaries: list[ItemSummary] = []
        for item in cleaned:
            summary = self._search_item(item)
            summaries.append(summary)
            self._log_summary(summary)
        return summaries

    def _search_item(self, item: str) -> ItemSummary:
        """Search every document/chunk for a single item."""
        pattern = re.compile(re.escape(item), re.IGNORECASE)
        matches: list[ItemMatch] = []
        context_parts: list[str] = []
        related_images: list[Path] = []

        for doc in self.documents:
            if doc.error:
                continue

            item_norm = _normalize(item)
            item_slug = _slugify(item)

            if doc.image_paths:
                name_norm = _normalize(doc.document_name)
                name_slug = _slugify(Path(doc.document_name).stem)
                if (
                    item_norm in name_norm
                    or item_slug in name_slug
                    or name_slug in item_slug
                ):
                    related_images.extend(doc.image_paths)

            for chunk in doc.chunks:
                for match in pattern.finditer(chunk.text):
                    location = chunk.location_label()
                    snippet = _snippet(chunk.text, match.start(), match.end())
                    matches.append(
                        ItemMatch(
                            item=item,
                            document_name=doc.document_name,
                            document_path=doc.path,
                            location=location,
                            snippet=snippet,
                            page_number=chunk.page_number,
                            line_number=chunk.line_number,
                        )
                    )
                    context_parts.append(
                        f"[{doc.document_name} | {location}]\n{chunk.text.strip()}"
                    )

            if any(m.document_path == doc.path for m in matches):
                related_images.extend(
                    self._related_sibling_images(
                        doc.path, item_norm=item_norm, item_slug=item_slug
                    )
                )
                matched_pages = {
                    m.page_number for m in matches if m.document_path == doc.path and m.page_number
                }
                for image in doc.image_paths:
                    page_num = self._page_num_from_image(image, fallback=0)
                    if matched_pages and page_num in matched_pages:
                        related_images.append(image)
                    elif not matched_pages and image not in related_images:
                        related_images.append(image)

        unique_images: list[Path] = []
        seen: set[Path] = set()
        for image in related_images:
            resolved = image.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_images.append(image)

        # Limit related page dumps for large scanned catalogs.
        if len(unique_images) > 6:
            unique_images = unique_images[:6]

        return ItemSummary(
            item=item,
            matches=matches,
            related_images=unique_images,
            context_text="\n\n".join(context_parts[:12]),
        )

    @staticmethod
    def _related_sibling_images(
        document_path: Path,
        *,
        item_norm: str,
        item_slug: str,
    ) -> list[Path]:
        """Collect nearby images whose names relate to the selected item."""
        parent = document_path.parent
        if not parent.exists():
            return []
        images: list[Path] = []
        for child in parent.iterdir():
            if not child.is_file():
                continue
            if child.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            stem_norm = _normalize(child.stem)
            stem_slug = _slugify(child.stem)
            if item_norm in stem_norm or item_slug in stem_slug or stem_slug in item_slug:
                images.append(child)
        return images

    @staticmethod
    def _log_summary(summary: ItemSummary) -> None:
        """Print/log a console-friendly breakdown for one item."""
        if not summary.matches:
            logger.info("Item '%s' was not found in any document.", summary.item)
            print(f"Item '{summary.item}' was not found in any document.")
            return

        print(f"\nItem '{summary.item}' — {len(summary.matches)} match(es):")
        for match in summary.matches:
            message = (
                f"Item '{match.item}' found in document '{match.document_name}' "
                f"on {match.location}"
            )
            logger.info(message)
            print(f"  • {message}")
            if match.snippet:
                print(f"    snippet: {match.snippet}")

        if summary.related_images:
            print(f"  Related images: {len(summary.related_images)}")
            for image in summary.related_images:
                print(f"    - {image.name}")

    def prompt_for_items(
        self,
        preselected: str | None = None,
        *,
        use_all: bool = False,
    ) -> list[str]:
        """Choose items based on content already read from ``inputs/``.

        Flow:
          1. Detect candidates from parsed/enriched documents.
          2. If ``--items`` / ``preselected`` was provided, use that.
          3. If ``use_all``, take every detected candidate.
          4. Otherwise show a numbered menu from the files that were read.

        Args:
            preselected: Optional CLI-provided selection (names, numbers, or ranges).
            use_all: Select every detected candidate without prompting.

        Returns:
            List of item/product/category names.
        """
        candidates = self.detect_candidates()
        if not candidates:
            print(
                "\nNo products detected from text patterns yet — "
                "trying AI extraction from the read pages/images…"
            )
            candidates = self.extract_candidates_with_ai()

        if not candidates:
            if preselected:
                raw = preselected.strip()
                if not raw:
                    return []
                return [part.strip() for part in raw.split(",") if part.strip()]
            print(
                "\nNo products/items were detected in the files under inputs/.\n"
                "You can still type names manually (comma-separated)."
            )
            raw = input("Items: ").strip()
            if not raw:
                logger.warning("No items entered.")
                return []
            return [part.strip() for part in raw.split(",") if part.strip()]

        if preselected:
            raw = preselected.strip()
            if not raw:
                return []
            selected = _resolve_item_selection(raw, candidates)
            print(f"\nSelected {len(selected)} product(s): " + "; ".join(selected))
            return selected

        print(f"\n=== Items/products detected from inputs ({len(candidates)}) ===")
        for idx, candidate in enumerate(candidates, start=1):
            where = f"{candidate.source_document} · {candidate.location}"
            print(f"  {idx}. [{candidate.kind}] {candidate.name}  ({where})")

        if use_all:
            selected = [c.name for c in candidates]
            print(f"\nUsing all {len(selected)} detected item(s).")
            return selected

        print(
            "\nSelect from the list above (these came from files already read).\n"
            "Enter numbers (e.g. 1,3 or 1-7,9), product names, 'all', or a mix.\n"
            f"Valid numbers are 1..{len(candidates)}."
        )
        raw = input("Selection: ").strip()
        if not raw:
            logger.warning("No selection entered.")
            return []
        selected = _resolve_item_selection(raw, candidates)
        print(f"\nSelected {len(selected)} product(s): " + "; ".join(selected))
        return selected
