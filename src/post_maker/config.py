"""Application configuration loaded from environment / `.env`."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root: post_maker/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS_DIR = PROJECT_ROOT / "inputs"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOCX_EXTENSIONS = {".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_EXTENSIONS = (
    SUPPORTED_TEXT_EXTENSIONS
    | SUPPORTED_PDF_EXTENSIONS
    | SUPPORTED_DOCX_EXTENSIONS
    | SUPPORTED_IMAGE_EXTENSIONS
)

# Platform aspect ratios as (width, height) in pixels.
PLATFORM_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "instagram": (1080, 1080),  # 1:1
    "facebook": (1080, 1080),  # 1:1
    "x": (1200, 675),  # 16:9
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for document parsing and Vertex AI generation."""

    project_root: Path = PROJECT_ROOT
    inputs_dir: Path = DEFAULT_INPUTS_DIR
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    google_genai_use_vertexai: bool = True
    gemini_model: str = "gemini-2.5-flash-lite"
    # Cheapest Vertex image model: Nano Banana 2 Lite
    image_model: str = "gemini-3.1-flash-lite-image"
    default_word_count: int = 50
    default_image_count: int = 2
    platforms: tuple[str, ...] = ("instagram", "facebook", "x")
    credentials_path: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict, repr=False)


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from `.env` and process environment variables.

    Args:
        env_file: Optional path to a `.env` file. Defaults to project root `.env`.

    Returns:
        Populated :class:`Settings` instance.
    """
    dotenv_path = env_file or (PROJECT_ROOT / ".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
        logger.debug("Loaded environment from %s", dotenv_path)
    else:
        logger.warning("No .env file found at %s; using process environment only.", dotenv_path)

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or None
    if credentials:
        # Ensure Vertex/google-auth clients see the same path.
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", credentials)

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
    model = (
        os.getenv("BU_GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "gemini-2.5-flash-lite"
    ).strip()
    image_model = (
        os.getenv("BU_IMAGE_MODEL")
        or os.getenv("GEMINI_IMAGE_MODEL")
        or "gemini-3.1-flash-lite-image"  # Nano Banana 2 Lite (cheapest)
    ).strip()
    try:
        default_image_count = int(os.getenv("DEFAULT_IMAGE_COUNT", "2"))
    except ValueError:
        default_image_count = 2

    if use_vertex and not project:
        logger.warning(
            "GOOGLE_GENAI_USE_VERTEXAI is enabled but GOOGLE_CLOUD_PROJECT is empty."
        )

    return Settings(
        google_cloud_project=project,
        google_cloud_location=location,
        google_genai_use_vertexai=use_vertex,
        gemini_model=model,
        image_model=image_model,
        default_image_count=max(1, default_image_count),
        credentials_path=credentials,
    )


def resolve_inputs_dir(subdir: str | None = None, settings: Settings | None = None) -> Path:
    """Resolve the inputs directory, optionally under a user subfolder.

    Args:
        subdir: Optional relative path under ``inputs/`` (e.g. ``catalog_2026``)
            or an absolute path. Also accepts values like ``inputs/catalog_2026``.
        settings: Optional settings override.

    Returns:
        Absolute path to the inputs directory to scan.

    Raises:
        FileNotFoundError: If the resolved directory does not exist.
    """
    cfg = settings or load_settings()
    base = cfg.inputs_dir

    if not subdir:
        if not base.exists():
            raise FileNotFoundError(f"Inputs directory does not exist: {base}")
        return base.resolve()

    candidate = Path(subdir).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    elif candidate.parts and candidate.parts[0] == "inputs":
        resolved = cfg.project_root / candidate
    else:
        resolved = base / candidate

    if not resolved.exists():
        raise FileNotFoundError(f"Inputs path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Inputs path is not a directory: {resolved}")
    return resolved.resolve()
