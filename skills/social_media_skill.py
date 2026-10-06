"""Social media skill: one campaign with multi-product images + platform posts."""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from post_maker.config import PLATFORM_ASPECT_RATIOS, Settings, load_settings
from post_maker.image_style import (
    DEFAULT_IMAGE_COUNT,
    DEFAULT_IMAGE_STYLE,
    DEFAULT_NEGATIVE_PROMPT,
    compose_image_generation_prompt,
    fill_product_category,
    is_blank_negative_prompt,
)
from post_maker.models import GeneratedAsset, ItemSummary, PlatformPost, TokenUsageTracker
from post_maker.vertex_client import generate_image, generate_text

logger = logging.getLogger(__name__)

DEFAULT_CONTENT_DIRECTION = "Introduce these product(s) or series"
MIN_HASHTAGS = 5

SYSTEM_INSTRUCTION = (
    "You are an expert social media copywriter for a multi-product marketing campaign. "
    "Write concise, engaging posts with a clear CTA. Ground the copy in the provided "
    "text context AND any attached catalog/product images. Do not invent products or "
    "facts that are not supported by the materials. "
    f"End every post with at least {MIN_HASHTAGS} searchable hashtags "
    "(#keyword form), on their own final line."
)


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate paths while preserving order."""
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "_", slug).strip("_").lower()
    return slug or "campaign"


def _count_words(text: str) -> int:
    """Count whitespace-separated words in body text (excluding hashtags)."""
    body = re.sub(r"#\w+", "", text)
    return len(re.findall(r"\b\w+\b", body, flags=re.UNICODE))


def _extract_hashtags(text: str) -> list[str]:
    """Return unique hashtags found in ``text`` (with leading #)."""
    found: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"#([\w]+)", text, flags=re.UNICODE):
        tag = f"#{match}"
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            found.append(tag)
    return found


def _hashtagify(token: str) -> str:
    """Convert a raw token into a searchable #keyword."""
    cleaned = re.sub(r"[^\w]+", "", token, flags=re.UNICODE)
    return f"#{cleaned}" if cleaned else ""


def ensure_min_hashtags(
    text: str,
    *,
    products: list[str],
    platform: str,
    content_direction: str,
    minimum: int = MIN_HASHTAGS,
) -> str:
    """Ensure the post ends with at least ``minimum`` searchable hashtags."""
    existing = _extract_hashtags(text)
    if len(existing) >= minimum:
        return text.rstrip()

    body = re.sub(r"(?:\s*#\w+)+\s*$", "", text).rstrip()
    seeds: list[str] = []
    for product in products:
        seeds.extend(re.findall(r"[\w]+", product, flags=re.UNICODE))
    seeds.extend(re.findall(r"[\w]+", content_direction, flags=re.UNICODE))
    seeds.extend([platform, "campaign", "collection", "series", "newarrival", "shopnow"])

    extras: list[str] = []
    seen = {tag.lower() for tag in existing}
    for seed in seeds:
        if len(seed) < 2:
            continue
        tag = _hashtagify(seed)
        if not tag or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        extras.append(tag)
        if len(existing) + len(extras) >= minimum:
            break

    while len(existing) + len(extras) < minimum:
        filler = f"#keyword{len(existing) + len(extras) + 1}"
        if filler.lower() not in seen:
            extras.append(filler)
            seen.add(filler.lower())

    return f"{body}\n\n{' '.join(existing + extras)}".strip()


def prompt_for_content_direction(
    *,
    preselected: str | None = None,
    default: str = DEFAULT_CONTENT_DIRECTION,
) -> str:
    """Ask for the creative direction used when generating campaign posts."""
    if preselected and preselected.strip():
        return preselected.strip()

    print("\nBefore generating, set the content direction for this campaign.")
    print(f"Default: {default}")
    raw = input("Content direction: ").strip()
    direction = raw or default
    logger.info("Content direction: %s", direction)
    print(f"Using content direction: {direction}")
    return direction


def prompt_for_image_count(
    *,
    preselected: int | None = None,
    default: int = DEFAULT_IMAGE_COUNT,
) -> int:
    """Ask how many campaign images to generate."""
    if preselected is not None and preselected > 0:
        return preselected

    print(f"\nHow many campaign images should be generated? Default: {default}")
    raw = input(f"Image count [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"  Invalid number; using default {default}.")
        return default
    if value < 1:
        print(f"  Must be >= 1; using default {default}.")
        return default
    return value


def prompt_for_image_style(
    *,
    preselected: str | None = None,
    style_file: Path | None = None,
    default: str = DEFAULT_IMAGE_STYLE,
) -> str:
    """Ask for / load the image style prompt used for campaign image generation."""
    if style_file is not None:
        text = style_file.read_text(encoding="utf-8")
        print(f"Loaded image style from {style_file}")
        return text

    if preselected and preselected.strip():
        return preselected.strip()

    print("\nImage style prompt for the campaign visuals.")
    print("Press Enter to use the default Japanese boutique collection style.")
    print("(Or paste a custom style. Type END on its own line to finish a multi-line paste.)")
    first = input("Image style [default]: ").strip()
    if not first:
        print("Using default Japanese boutique image style.")
        return default
    if first.upper() == "END":
        return default

    lines = [first]
    if first.upper() != "DEFAULT":
        print("  (Continue typing. Enter END on a new line when finished.)")
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    style = "\n".join(lines).strip()
    return style or default


def prompt_for_negative_prompt(
    *,
    preselected: str | None = None,
    default: str = DEFAULT_NEGATIVE_PROMPT,
) -> str:
    """Ask for an optional negative prompt after the image style step.

    Args:
        preselected: Optional CLI-provided negative prompt.
        default: Default when the user presses Enter (``.`` = none).

    Returns:
        Negative prompt string. ``.`` / blank means no exclusions.
    """
    if preselected is not None:
        value = preselected.strip() or default
        if is_blank_negative_prompt(value):
            print("Negative prompt: none")
        else:
            print(f"Negative prompt: {value}")
        return value

    print("\nNegative prompt (elements to SKIP / exclude from the image).")
    print("Default: .  (none — do not exclude anything extra)")
    raw = input("Negative prompt [.]: ").strip()
    value = raw if raw else default
    if is_blank_negative_prompt(value):
        print("Negative prompt: none")
        return DEFAULT_NEGATIVE_PROMPT
    print(f"Negative prompt: {value}")
    return value


def _campaign_slug(products: list[str]) -> str:
    """Build a short campaign folder slug from selected products."""
    parts = [_slugify(name) for name in products[:3]]
    base = "_".join(part for part in parts if part) or "campaign"
    if len(products) > 3:
        base = f"{base}_plus{len(products) - 3}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"campaign_{base}_{stamp}"


def bytes_to_jpeg(image_bytes: bytes, *, quality: int = 92) -> bytes:
    """Convert arbitrary image bytes to JPEG bytes."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()


class SocialMediaSkill:
    """Generate one multi-product campaign: N styled images + platform posts."""

    def __init__(
        self,
        settings: Settings | None = None,
        tracker: TokenUsageTracker | None = None,
    ) -> None:
        """
        Args:
            settings: Optional application settings (Vertex models, output paths).
            tracker: Optional shared token-usage tracker for this run.
        """
        self.settings = settings or load_settings()
        self.tracker = tracker if tracker is not None else TokenUsageTracker()

    def generate_campaign(
        self,
        summaries: list[ItemSummary],
        *,
        content_direction: str = DEFAULT_CONTENT_DIRECTION,
        image_style: str = DEFAULT_IMAGE_STYLE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        image_count: int = DEFAULT_IMAGE_COUNT,
        word_count: int | None = None,
        platforms: list[str] | None = None,
        outputs_dir: Path | None = None,
    ) -> GeneratedAsset | None:
        """Generate a single campaign for all selected products.

        Args:
            summaries: Item search results for every selected product.
            content_direction: Copy direction for social posts.
            image_style: Art-direction template (may include ``[PRODUCT_CATEGORY]``).
            negative_prompt: Elements to exclude from images (``.`` = none).
            image_count: Number of campaign images to generate.
            word_count: Approximate post body length.
            platforms: Target platforms.
            outputs_dir: Root outputs directory.

        Returns:
            :class:`GeneratedAsset` for the campaign, or ``None`` if nothing usable.
        """
        usable = [s for s in summaries if s.found or s.related_images]
        if not usable:
            logger.warning("No usable products for campaign generation.")
            return None

        products = [s.item for s in usable]
        target_words = word_count or self.settings.default_word_count
        target_platforms = [
            p.lower().strip()
            for p in (platforms or list(self.settings.platforms))
            if p.strip()
        ]
        out_root = outputs_dir or self.settings.outputs_dir
        campaign_dir = out_root / _campaign_slug(products)
        campaign_dir.mkdir(parents=True, exist_ok=True)

        filled_style = fill_product_category(image_style, products)
        style_path = campaign_dir / "image_style.txt"
        style_path.write_text(filled_style.rstrip() + "\n", encoding="utf-8")

        negative_path = campaign_dir / "negative_prompt.txt"
        negative_value = (
            DEFAULT_NEGATIVE_PROMPT
            if is_blank_negative_prompt(negative_prompt)
            else negative_prompt.strip()
        )
        negative_path.write_text(negative_value + "\n", encoding="utf-8")

        for platform in target_platforms:
            (campaign_dir / platform).mkdir(parents=True, exist_ok=True)

        reference_images = _unique_paths(
            [path for summary in usable for path in summary.related_images if path.exists()]
        )
        # Cap reference uploads to keep requests manageable.
        reference_images = reference_images[:6]
        context = "\n\n".join(
            f"### {summary.item}\n{summary.context_text.strip()}"
            for summary in usable
            if summary.context_text.strip()
        )

        print(
            f"\nGenerating campaign for {len(products)} product(s) → "
            f"{image_count} image(s) under {campaign_dir.name}/{{platform}}/"
        )
        print("Products (allowlist): " + "; ".join(products))
        print("Platforms: " + ", ".join(target_platforms))
        if is_blank_negative_prompt(negative_value):
            print("Negative prompt: none")
        else:
            print(f"Negative prompt: {negative_value}")

        image_paths = self._generate_campaign_images(
            products=products,
            image_style=image_style,
            negative_prompt=negative_value,
            reference_images=reference_images,
            image_count=image_count,
            campaign_dir=campaign_dir,
            platforms=target_platforms,
        )

        posts = self._generate_campaign_posts(
            products=products,
            context=context,
            word_count=target_words,
            platforms=target_platforms,
            content_direction=content_direction,
            reference_images=reference_images,
        )

        asset = GeneratedAsset(
            campaign_name=campaign_dir.name,
            products=products,
            output_dir=campaign_dir,
            style_prompt_path=style_path,
            content_direction=content_direction,
            image_style=filled_style,
            images=image_paths,
        )

        overview_lines = [
            f"# Campaign: {campaign_dir.name}",
            f"# Products: {', '.join(products)}",
            f"# Content direction: {content_direction}",
            f"# Negative prompt: {negative_value}",
            f"# Target length: ~{target_words} words",
            f"# Images: {image_count}",
            f"# Platforms: {', '.join(target_platforms)}",
            "",
        ]
        for post in posts:
            platform_dir = campaign_dir / post.platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            platform_path = platform_dir / "post.txt"
            platform_path.write_text(post.text + "\n", encoding="utf-8")
            asset.posts[post.platform] = platform_path
            overview_lines.extend(
                [
                    f"## {post.platform.upper()} ({post.word_count} words)",
                    post.text,
                    "",
                ]
            )
            logger.info("Wrote %s", platform_path)

        combined_path = campaign_dir / "campaign_overview.txt"
        combined_path.write_text("\n".join(overview_lines).rstrip() + "\n", encoding="utf-8")
        asset.combined_post_path = combined_path

        products_path = campaign_dir / "products.txt"
        products_path.write_text("\n".join(products) + "\n", encoding="utf-8")
        return asset

    def _generate_campaign_images(
        self,
        *,
        products: list[str],
        image_style: str,
        negative_prompt: str,
        reference_images: list[Path],
        image_count: int,
        campaign_dir: Path,
        platforms: list[str],
    ) -> list[Path]:
        """Generate N AI images once, then locally resize into each platform folder.

        Cost note:
            AI is called exactly ``image_count`` times (one set of masters).
            Instagram / Facebook / X variants are Pillow resizes only — no extra
            image-generation tokens.
        """
        written: list[Path] = []
        product_line = ", ".join(products)
        print(
            f"  Image plan: {image_count} AI master(s) → local resize for "
            f"{', '.join(platforms)} (no extra AI image calls per platform)."
        )

        source_dir = campaign_dir / "_source"
        source_dir.mkdir(parents=True, exist_ok=True)

        for index in range(1, image_count + 1):
            step = f"campaign_image:{index}"
            variation = (
                f"Variation request #{index} of {image_count}: "
                "Change ONLY camera angle, lighting, or staging layout from other variations. "
                "DO NOT add, remove, swap, or invent any products.\n"
                f"Selected products for this campaign: {product_line}\n"
                "RULE #1: Make every product EXACTLY the same as in the source images. "
                "If you are not sure, do NOT put that product — skip it.\n"
                "Do NOT invent products not in the attached source document.\n"
                "Critical scale reminder: keep real-world proportions; "
                "product must never look larger than its packaging.\n"
                "Return exactly one final social-media-ready photograph."
            )
            prompt = compose_image_generation_prompt(
                style_template=image_style,
                products=products,
                negative_prompt=negative_prompt,
                variation_note=variation,
            )
            prompt_path = source_dir / f"prompt_image_{index}.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")

            print(f"  Generating AI master {index}/{image_count}…")
            result = generate_image(
                prompt,
                settings=self.settings,
                step=step,
                tracker=self.tracker,
                image_paths=reference_images or None,
            )
            if not result.image_bytes:
                logger.error("Failed to generate campaign image %d.", index)
                print(f"  *** Failed to generate image {index} ***")
                continue

            jpeg_bytes = bytes_to_jpeg(result.image_bytes)
            print(
                f"  AI master {index}/{image_count} ready "
                f"({result.usage.total_tokens} tokens) → resizing for platforms…"
            )

            source_path = source_dir / f"image_{index}.jpg"
            source_path.write_bytes(jpeg_bytes)
            written.append(source_path)
            logger.info("Wrote AI master %s", source_path)

            with Image.open(io.BytesIO(jpeg_bytes)) as img:
                rgb = img.convert("RGB")
                for platform in platforms:
                    platform_dir = campaign_dir / platform
                    platform_dir.mkdir(parents=True, exist_ok=True)
                    size = PLATFORM_ASPECT_RATIOS.get(platform) or PLATFORM_ASPECT_RATIOS["instagram"]
                    framed = self._fit_cover(rgb, size)
                    out_path = platform_dir / f"image_{index}.jpg"
                    framed.save(out_path, format="JPEG", quality=92, optimize=True)
                    written.append(out_path)
                    logger.info("Resized (local, 0 tokens) → %s", out_path)

        if not written:
            logger.warning("No campaign images were generated; writing placeholders.")
            return self._write_placeholders(campaign_dir, platforms)
        return written

    def _generate_campaign_posts(
        self,
        *,
        products: list[str],
        context: str,
        word_count: int,
        platforms: list[str],
        content_direction: str,
        reference_images: list[Path],
    ) -> list[PlatformPost]:
        """Generate one set of platform posts covering all selected products."""
        results: list[PlatformPost] = []
        images = reference_images[:3]
        product_line = ", ".join(products)

        for platform in platforms:
            step = f"campaign_post:{platform}"
            prompt = self._build_campaign_prompt(
                products=products,
                context=context,
                word_count=word_count,
                platform=platform,
                content_direction=content_direction,
                image_count=len(images),
            )
            result = generate_text(
                prompt,
                settings=self.settings,
                system_instruction=SYSTEM_INSTRUCTION,
                step=step,
                tracker=self.tracker,
                image_paths=images or None,
            )
            text = result.text
            if not text:
                text = (
                    f"{content_direction.rstrip('.')}. "
                    f"Discover this collection: {product_line}. "
                    f"Curated for {platform.title()} — shop the series today."
                )
            text = ensure_min_hashtags(
                text,
                products=products,
                platform=platform,
                content_direction=content_direction,
            )
            results.append(
                PlatformPost(
                    platform=platform,
                    text=text,
                    word_count=_count_words(text),
                )
            )
        return results

    @staticmethod
    def _build_campaign_prompt(
        *,
        products: list[str],
        context: str,
        word_count: int,
        platform: str,
        content_direction: str,
        image_count: int,
    ) -> str:
        """Compose the Gemini prompt for one campaign platform post."""
        platform_notes = {
            "instagram": "Warm, visual, emoji-light; suitable for Instagram caption.",
            "facebook": "Friendly and conversational; suitable for a Facebook feed post.",
            "x": "Punchy and concise; suitable for X/Twitter.",
        }
        note = platform_notes.get(platform, f"Tailor the tone for {platform}.")
        context_block = context.strip() or "(No additional document text context available.)"
        image_note = (
            f"{image_count} related catalog/product image(s) are attached."
            if image_count
            else "No images attached; rely on the text/document context."
        )
        product_line = ", ".join(products)
        return (
            f"Write ONE social media campaign post for {platform} that covers "
            f"this multi-product collection together (not separate posts).\n"
            f"Selected products: {product_line}\n"
            f"Content direction: {content_direction}\n"
            f"Target body length: about {word_count} words (excluding hashtags).\n"
            f"Style note: {note}\n"
            f"Visual inputs: {image_note}\n\n"
            f"Document context:\n{context_block}\n\n"
            f"Requirements:\n"
            f"- Treat this as one marketing campaign featuring all selected products.\n"
            f"- Follow the content direction closely.\n"
            f"- Base claims only on provided text/images.\n"
            f"- After the body, add a blank line, then at least {MIN_HASHTAGS} "
            f"searchable hashtags.\n"
            "Return only the post text, with no preamble or markdown fences."
        )

    @staticmethod
    def _fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Center-crop/resize an image to exactly ``size`` (cover fit)."""
        target_w, target_h = size
        src_w, src_h = image.size
        if src_w == 0 or src_h == 0:
            return Image.new("RGB", size, color=(240, 240, 240))

        scale = max(target_w / src_w, target_h / src_h)
        resized = image.resize(
            (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
            Image.Resampling.LANCZOS,
        )
        left = max(0, (resized.width - target_w) // 2)
        top = max(0, (resized.height - target_h) // 2)
        return resized.crop((left, top, left + target_w, top + target_h))

    def _write_placeholders(
        self,
        campaign_dir: Path,
        platforms: list[str],
    ) -> list[Path]:
        """Create solid-color JPEG placeholders under each platform folder."""
        written: list[Path] = []
        for platform in platforms:
            platform_dir = campaign_dir / platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            size = PLATFORM_ASPECT_RATIOS.get(platform, (1080, 1080))
            framed = Image.new("RGB", size, color=(32, 56, 88))
            out_path = platform_dir / "image_1.jpg"
            framed.save(out_path, format="JPEG", quality=90, optimize=True)
            written.append(out_path)
            logger.info("Wrote placeholder %s", out_path)
        return written
