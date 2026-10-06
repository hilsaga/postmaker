"""Shared data models for parsed documents and generated content."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TextChunk:
    """A contiguous text segment with location metadata."""

    text: str
    page_number: int | None = None
    section: str | None = None
    line_number: int | None = None

    def location_label(self) -> str:
        """Return a human-readable location string."""
        if self.page_number is not None:
            return f"Page {self.page_number}"
        if self.section:
            return f"Section '{self.section}'"
        if self.line_number is not None:
            return f"Line {self.line_number}"
        return "Unknown location"


@dataclass
class ParsedDocument:
    """Parsed representation of a single input file."""

    path: Path
    document_name: str
    file_type: str
    chunks: list[TextChunk] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def full_text(self) -> str:
        """Concatenate all chunk text for whole-document search."""
        return "\n".join(chunk.text for chunk in self.chunks if chunk.text)


@dataclass
class ItemMatch:
    """A match of a user-selected item inside a document."""

    item: str
    document_name: str
    document_path: Path
    location: str
    snippet: str
    page_number: int | None = None
    line_number: int | None = None
    category_hint: str | None = None


@dataclass
class DetectedCandidate:
    """An item/product/category discovered from input documents."""

    name: str
    kind: str = "item"  # item | product | category
    source_document: str = ""
    location: str = ""
    snippet: str = ""


@dataclass
class ItemSummary:
    """Aggregated findings for one selected item/product/category."""

    item: str
    matches: list[ItemMatch] = field(default_factory=list)
    related_images: list[Path] = field(default_factory=list)
    context_text: str = ""

    @property
    def found(self) -> bool:
        """Whether at least one textual match was found."""
        return bool(self.matches)


@dataclass
class PlatformPost:
    """Generated social post text for a single platform."""

    platform: str
    text: str
    word_count: int


@dataclass
class GeneratedAsset:
    """Paths to artifacts written for one marketing campaign."""

    campaign_name: str
    products: list[str]
    output_dir: Path
    posts: dict[str, Path] = field(default_factory=dict)
    images: list[Path] = field(default_factory=list)
    combined_post_path: Path | None = None
    style_prompt_path: Path | None = None
    content_direction: str = ""
    image_style: str = ""

    # Backward-compatible alias used by older call sites / prints.
    @property
    def product_name(self) -> str:
        """Return a display label for the campaign products."""
        return self.campaign_name



@dataclass
class TokenUsage:
    """Token counts for a single model call."""

    step: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0

    def __post_init__(self) -> None:
        """Ensure ``total_tokens`` covers input + output + thoughts."""
        computed = self.input_tokens + self.output_tokens + self.thoughts_tokens
        if self.total_tokens < computed:
            self.total_tokens = computed

    def as_line(self) -> str:
        """Format a single-line summary for logging/printing."""
        extra = ""
        if self.thoughts_tokens:
            extra = f" | thoughts={self.thoughts_tokens}"
        return (
            f"[{self.step}] tokens — "
            f"input={self.input_tokens}, output={self.output_tokens}, "
            f"total={self.total_tokens}{extra}"
        )


@dataclass
class TokenUsageTracker:
    """Accumulate and report token usage across generation steps."""

    steps: list[TokenUsage] = field(default_factory=list)

    def add(self, usage: TokenUsage) -> TokenUsage:
        """Record one step and return it for chaining."""
        # Re-normalize in case callers constructed TokenUsage without __post_init__ path.
        computed = usage.input_tokens + usage.output_tokens + usage.thoughts_tokens
        if usage.total_tokens < computed:
            usage.total_tokens = computed
        self.steps.append(usage)
        return usage

    @property
    def input_tokens(self) -> int:
        """Sum of prompt/input tokens across all steps."""
        return sum(step.input_tokens for step in self.steps)

    @property
    def output_tokens(self) -> int:
        """Sum of candidate/output tokens across all steps."""
        return sum(step.output_tokens for step in self.steps)

    @property
    def thoughts_tokens(self) -> int:
        """Sum of thoughts tokens across all steps."""
        return sum(step.thoughts_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        """Authoritative run total: sum of per-step totals (normalized)."""
        return sum(step.total_tokens for step in self.steps)

    def _category(self, step_name: str) -> str:
        """Map a step label to a billing/category bucket."""
        name = step_name.lower()
        if name.startswith("vision:") or name.startswith("visible_inventory") or name.startswith(
            "fidelity_check:"
        ):
            return "vision_ocr"
        if name.startswith("campaign_image:") or name.startswith("generate_image"):
            return "image_generation"
        if name.startswith("campaign_post:") or name.startswith("post:"):
            return "text_posts"
        return "other"

    def category_totals(self) -> dict[str, TokenUsage]:
        """Aggregate token usage by category."""
        buckets: dict[str, TokenUsage] = {}
        for step in self.steps:
            key = self._category(step.step)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = TokenUsage(step=key)
                buckets[key] = bucket
            bucket.input_tokens += step.input_tokens
            bucket.output_tokens += step.output_tokens
            bucket.thoughts_tokens += step.thoughts_tokens
            bucket.total_tokens += step.total_tokens
        return buckets

    def summary_lines(self) -> list[str]:
        """Build human-readable per-step, category, and total summary lines."""
        lines = ["=== Token usage ==="]
        if not self.steps:
            lines.append("No model calls recorded.")
            return lines

        lines.append("-- Per step --")
        for step in self.steps:
            lines.append(f"  {step.as_line()}")

        lines.append("-- By category --")
        for key, usage in self.category_totals().items():
            lines.append(f"  {usage.as_line()}")

        total = TokenUsage(
            step="TOTAL",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            thoughts_tokens=self.thoughts_tokens,
        )
        lines.append("-- Run total --")
        lines.append(f"  {total.as_line()}")
        lines.append(
            "  note: platform image crops are local Pillow resizes (0 AI tokens)."
        )
        return lines

    def print_summary(self) -> None:
        """Print the token usage summary to the console."""
        print()
        for line in self.summary_lines():
            print(line)

    def write_report(self, path: Path) -> Path:
        """Persist the token usage summary to ``path``.

        Args:
            path: Destination text file.

        Returns:
            The written path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.summary_lines()) + "\n", encoding="utf-8")
        return path

