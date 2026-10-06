"""Vertex AI / Gemini client helpers using google-genai."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai

from post_maker.config import Settings, load_settings
from post_maker.models import TokenUsage, TokenUsageTracker

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Text response plus token usage for one model call."""

    text: str
    usage: TokenUsage


@dataclass
class ImageGenerationResult:
    """Image bytes plus token usage for one image generation call."""

    image_bytes: bytes
    mime_type: str
    usage: TokenUsage
    text: str = ""
    error: str = ""
    is_rate_limited: bool = False


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect Vertex/Gemini quota / rate-limit failures."""
    message = str(exc).upper()
    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "TOO MANY REQUESTS" in message
        or "QUOTA" in message
    )


def extract_token_usage(response: Any, *, step: str) -> TokenUsage:
    """Parse ``usage_metadata`` from a generate_content response.

    Args:
        response: Google GenAI ``GenerateContentResponse`` (or compatible object).
        step: Human-readable label for this generation step.

    Returns:
        :class:`TokenUsage` with input/output/total counts (zeros if unavailable).
        ``total_tokens`` is normalized to at least input + output + thoughts.
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        logger.warning("No usage_metadata on response for step '%s'.", step)
        return TokenUsage(step=step)

    input_tokens = int(getattr(meta, "prompt_token_count", None) or 0)
    output_tokens = int(getattr(meta, "candidates_token_count", None) or 0)
    thoughts_tokens = int(getattr(meta, "thoughts_token_count", None) or 0)
    total_tokens = int(getattr(meta, "total_token_count", None) or 0)
    computed = input_tokens + output_tokens + thoughts_tokens
    if total_tokens < computed:
        total_tokens = computed

    return TokenUsage(
        step=step,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        thoughts_tokens=thoughts_tokens,
    )


@lru_cache(maxsize=1)
def get_genai_client(settings_key: tuple[str, str, bool, str | None] | None = None) -> genai.Client:
    """Create a cached Google GenAI client configured for Vertex AI.

    Args:
        settings_key: Optional cache key tuple
            ``(project, location, use_vertexai, credentials_path)``.
            When omitted, settings are loaded from the environment.

    Returns:
        Configured :class:`google.genai.Client`.
    """
    settings = load_settings()
    if settings_key is None:
        settings_key = (
            settings.google_cloud_project,
            settings.google_cloud_location,
            settings.google_genai_use_vertexai,
            settings.credentials_path,
        )

    project, location, use_vertex, _credentials = settings_key
    if use_vertex:
        logger.info(
            "Initializing Vertex AI GenAI client (project=%s, location=%s)",
            project,
            location,
        )
        return genai.Client(
            vertexai=True,
            project=project or None,
            location=location or "global",
        )

    logger.info("Initializing Google GenAI client with API key mode")
    return genai.Client()


def _image_mime(path: Path) -> str:
    """Return a MIME type for a local image path."""
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def generate_text(
    prompt: str,
    *,
    settings: Settings | None = None,
    system_instruction: str | None = None,
    step: str = "generate_text",
    tracker: TokenUsageTracker | None = None,
    image_paths: list[Path] | None = None,
) -> GenerationResult:
    """Generate text via Gemini on Vertex AI and record token usage.

    Args:
        prompt: User prompt content.
        settings: Optional settings override.
        system_instruction: Optional system instruction for the model.
        step: Label used in token-usage logs for this call.
        tracker: Optional tracker that accumulates usage across steps.
        image_paths: Optional local image files to include (vision / OCR).

    Returns:
        :class:`GenerationResult` with response text and token counts.
        On failure, ``text`` is empty and token counts are zero.
    """
    cfg = settings or load_settings()
    client = get_genai_client(
        (
            cfg.google_cloud_project,
            cfg.google_cloud_location,
            cfg.google_genai_use_vertexai,
            cfg.credentials_path,
        )
    )

    from google.genai import types

    config: Any = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)

    contents: Any = prompt
    if image_paths:
        parts: list[Any] = []
        for image_path in image_paths:
            path = Path(image_path)
            if not path.exists():
                logger.warning("Image missing for vision call: %s", path)
                continue
            parts.append(
                types.Part.from_bytes(data=path.read_bytes(), mime_type=_image_mime(path))
            )
        parts.append(prompt)
        contents = parts
        if len(parts) == 1:
            logger.error("No readable images for vision step '%s'.", step)
            usage = TokenUsage(step=step)
            if tracker is not None:
                tracker.add(usage)
            return GenerationResult(text="", usage=usage)

    try:
        response = client.models.generate_content(
            model=cfg.gemini_model,
            contents=contents,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - surface API errors as logs
        logger.error("Vertex/Gemini generation failed (%s): %s", step, exc)
        usage = TokenUsage(step=step)
        if tracker is not None:
            tracker.add(usage)
        logger.info(usage.as_line())
        print(usage.as_line())
        return GenerationResult(text="", usage=usage)

    text = (getattr(response, "text", None) or "").strip()
    usage = extract_token_usage(response, step=step)
    if tracker is not None:
        tracker.add(usage)

    logger.info(usage.as_line())
    print(usage.as_line())
    return GenerationResult(text=text, usage=usage)


def generate_image(
    prompt: str,
    *,
    settings: Settings | None = None,
    step: str = "generate_image",
    tracker: TokenUsageTracker | None = None,
    image_paths: list[Path] | None = None,
    model: str | None = None,
) -> ImageGenerationResult:
    """Generate one campaign image via a Gemini image model on Vertex AI.

    Args:
        prompt: Image-generation style / art-direction prompt.
        settings: Optional settings override.
        step: Label used in token-usage logs for this call.
        tracker: Optional tracker that accumulates usage across steps.
        image_paths: Reference catalog/product images to keep products authentic.
        model: Optional image model override (defaults to ``settings.image_model``).

    Returns:
        :class:`ImageGenerationResult` with JPEG/PNG bytes and token usage.
        ``image_bytes`` is empty on failure.
    """
    from google.genai import types

    cfg = settings or load_settings()
    client = get_genai_client(
        (
            cfg.google_cloud_project,
            cfg.google_cloud_location,
            cfg.google_genai_use_vertexai,
            cfg.credentials_path,
        )
    )
    image_model = (model or cfg.image_model).strip()

    parts: list[Any] = []
    for image_path in image_paths or []:
        path = Path(image_path)
        if not path.exists():
            logger.warning("Reference image missing for image gen: %s", path)
            continue
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=_image_mime(path)))
    parts.append(prompt)

    try:
        response = client.models.generate_content(
            model=image_model,
            contents=parts,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as exc:  # noqa: BLE001
        rate_limited = _is_rate_limit_error(exc)
        logger.error("Vertex/Gemini image generation failed (%s): %s", step, exc)
        usage = TokenUsage(step=step)
        if tracker is not None:
            tracker.add(usage)
        logger.info(usage.as_line())
        print(usage.as_line())
        return ImageGenerationResult(
            image_bytes=b"",
            mime_type="image/jpeg",
            usage=usage,
            error=str(exc),
            is_rate_limited=rate_limited,
        )

    image_bytes = b""
    mime_type = "image/png"
    text_bits: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                image_bytes = bytes(inline.data)
                mime_type = getattr(inline, "mime_type", None) or "image/png"
            part_text = getattr(part, "text", None)
            if part_text:
                text_bits.append(part_text)

    usage = extract_token_usage(response, step=step)
    if tracker is not None:
        tracker.add(usage)
    logger.info(usage.as_line())
    print(usage.as_line())

    if not image_bytes:
        logger.error("Image generation returned no image bytes for step '%s'.", step)

    return ImageGenerationResult(
        image_bytes=image_bytes,
        mime_type=mime_type,
        usage=usage,
        text="\n".join(text_bits).strip(),
    )
