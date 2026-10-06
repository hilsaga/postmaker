#!/usr/bin/env python3
"""CLI entry point for the document parser and social media campaign generator."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure local packages resolve whether or not PYTHONPATH is set.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from post_maker.config import load_settings, resolve_inputs_dir  # noqa: E402
from post_maker.document_loader import (  # noqa: E402
    apply_pdf_page_selection,
    load_documents,
    parse_page_list,
    print_document_inventory,
    prompt_for_pdf_pages,
)
from post_maker.image_style import (  # noqa: E402
    DEFAULT_IMAGE_COUNT,
    DEFAULT_IMAGE_STYLE,
    DEFAULT_NEGATIVE_PROMPT,
)
from post_maker.models import TokenUsageTracker  # noqa: E402
from skills.information_skill import InformationSkill  # noqa: E402
from skills.social_media_skill import (  # noqa: E402
    DEFAULT_CONTENT_DIRECTION,
    SocialMediaSkill,
    prompt_for_content_direction,
    prompt_for_image_count,
    prompt_for_image_style,
    prompt_for_negative_prompt,
)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for console output.

    Args:
        verbose: Enable DEBUG-level logs when True.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Read inputs/, select PDF pages + products into one marketing campaign, "
            "then generate N collection images + social posts into outputs/."
        )
    )
    parser.add_argument(
        "--input-subdir",
        default=None,
        help=(
            "Optional subfolder under inputs/ to scan "
            "(e.g. catalog_2026 or inputs/catalog_2026)."
        ),
    )
    parser.add_argument(
        "--items",
        default=None,
        help=(
            "Items to include: names, numbers, or ranges (e.g. 1-7,9 or Product A). "
            "If omitted, items are detected from the files that were read."
        ),
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Use every item/product detected from the input files (no prompt).",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help=(
            "Comma-separated PDF page numbers to read (e.g. 1,2,5 or 1-3). "
            "If omitted and a PDF is present, you will be asked interactively."
        ),
    )
    parser.add_argument(
        "--direction",
        default=None,
        help=(
            "Content direction for campaign posts. "
            f"Default if omitted/blank: '{DEFAULT_CONTENT_DIRECTION}'."
        ),
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=None,
        help=f"Number of campaign images to generate (default: {DEFAULT_IMAGE_COUNT}).",
    )
    parser.add_argument(
        "--image-style",
        default=None,
        help="Custom image style prompt. If omitted, you can accept the default style.",
    )
    parser.add_argument(
        "--image-style-file",
        default=None,
        help="Path to a text file containing the image style prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help=(
            "Negative prompt: elements to skip/exclude in generated images. "
            f"Default if omitted/blank: '{DEFAULT_NEGATIVE_PROMPT}' (none)."
        ),
    )
    parser.add_argument(
        "--word-count",
        type=int,
        default=None,
        help="Approximate target word count for generated posts (default: 50).",
    )
    parser.add_argument(
        "--platforms",
        default="instagram,facebook,x",
        help="Comma-separated platforms: instagram,facebook,x (default: all).",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Attempt local OCR on image inputs when pytesseract is available.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only parse, enrich, and search; do not generate campaign assets.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end campaign pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger("post_maker")

    settings = load_settings()
    tracker = TokenUsageTracker()
    logger.info(
        "Vertex AI enabled=%s | project=%s | location=%s | text=%s | image=%s (Nano Banana 2 Lite)",
        settings.google_genai_use_vertexai,
        settings.google_cloud_project or "(unset)",
        settings.google_cloud_location,
        settings.gemini_model,
        settings.image_model,
    )

    try:
        inputs_dir = resolve_inputs_dir(args.input_subdir, settings)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Scanning inputs at %s", inputs_dir)
    documents = load_documents(inputs_dir, use_ocr=args.ocr, max_pages=None)
    if not documents:
        logger.error(
            "No documents to process. Add supported files to %s and retry.",
            inputs_dir,
        )
        return 1

    print_document_inventory(documents)

    ok_docs = [d for d in documents if not d.error]
    err_docs = [d for d in documents if d.error]
    logger.info("Parsed %d document(s) (%d with errors).", len(ok_docs), len(err_docs))
    for doc in err_docs:
        logger.warning("  %s: %s", doc.document_name, doc.error)

    preselected_pages = parse_page_list(args.pages) if args.pages else None
    page_selections = prompt_for_pdf_pages(documents, preselected=preselected_pages)
    for doc in documents:
        if doc.file_type != "pdf" or doc.error:
            continue
        pages = page_selections.get(doc.document_name) or [1]
        apply_pdf_page_selection(doc, pages)

    if page_selections:
        print_document_inventory(documents)

    info_skill = InformationSkill(documents, settings=settings, tracker=tracker)
    info_skill.enrich_scanned_documents()

    if any(d.metadata.get("text_source") == "vertex_vision" for d in documents):
        print_document_inventory(documents)

    items = info_skill.prompt_for_items(args.items, use_all=args.all_items)
    if not items:
        logger.error("No items selected; nothing to build a campaign from.")
        return 1

    summaries = info_skill.find_items(items)
    found_count = sum(1 for s in summaries if s.found or s.related_images)
    if found_count == 0:
        logger.warning("None of the selected items were found. Exiting.")
        tracker.print_summary()
        return 0

    print(
        f"\n=== Campaign selection ===\n"
        f"Products in this campaign ({found_count}): "
        + ", ".join(s.item for s in summaries if s.found or s.related_images)
    )

    if args.skip_generate:
        logger.info("Skipping campaign generation (--skip-generate).")
        tracker.print_summary()
        return 0

    content_direction = prompt_for_content_direction(preselected=args.direction)
    image_count = prompt_for_image_count(
        preselected=args.image_count,
        default=settings.default_image_count or DEFAULT_IMAGE_COUNT,
    )
    style_file = Path(args.image_style_file) if args.image_style_file else None
    if style_file is not None or args.image_style is not None:
        image_style = prompt_for_image_style(
            preselected=args.image_style,
            style_file=style_file,
        )
    elif args.direction is not None and args.image_count is not None:
        image_style = DEFAULT_IMAGE_STYLE
        print("Using default Japanese boutique image style.")
    else:
        image_style = prompt_for_image_style()

    if args.negative_prompt is not None:
        negative_prompt = prompt_for_negative_prompt(preselected=args.negative_prompt)
    elif args.direction is not None and args.image_count is not None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        print("Negative prompt: none")
    else:
        negative_prompt = prompt_for_negative_prompt()

    platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]
    social_skill = SocialMediaSkill(settings, tracker=tracker)
    asset = social_skill.generate_campaign(
        summaries,
        content_direction=content_direction,
        image_style=image_style,
        negative_prompt=negative_prompt,
        image_count=image_count,
        word_count=args.word_count,
        platforms=platforms,
        outputs_dir=settings.outputs_dir,
    )

    tracker.print_summary()
    for line in tracker.summary_lines():
        logger.info(line)

    if asset is None:
        logger.error("Campaign generation produced no assets.")
        return 1

    token_report = tracker.write_report(asset.output_dir / "token_usage.txt")
    print(f"\nToken report: {token_report}")

    print("\n=== Campaign generation complete ===")
    print(f"\nCampaign: {asset.campaign_name}")
    print(f"  Products: {', '.join(asset.products)}")
    print(f"  Output folder: {asset.output_dir}")
    print(
        f"  Tokens TOTAL: input={tracker.input_tokens}, "
        f"output={tracker.output_tokens}, total={tracker.total_tokens}"
    )
    if asset.style_prompt_path:
        print(f"  Style prompt: {asset.style_prompt_path.name}")
    if asset.combined_post_path:
        print(f"  Overview: {asset.combined_post_path.name}")
    for platform, path in asset.posts.items():
        platform_dir = path.parent
        print(f"  {platform}/")
        print(f"    - post: {path.name}")
        for image in sorted(platform_dir.glob("image_*.jpg")):
            print(f"    - image: {image.name}")

    logger.info(
        "Wrote campaign under %s | tokens TOTAL=%d (in=%d, out=%d, thoughts=%d)",
        asset.output_dir,
        tracker.total_tokens,
        tracker.input_tokens,
        tracker.output_tokens,
        tracker.thoughts_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
