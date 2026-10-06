"""Default campaign image style prompt and helpers."""

from __future__ import annotations

DEFAULT_IMAGE_COUNT = 2
DEFAULT_NEGATIVE_PROMPT = "."

DEFAULT_IMAGE_STYLE = """Role & Task:
You are an expert social media art director and commercial product photographer specializing in high-end Japanese visual merchandising. I have uploaded catalog PDF page image(s) and/or product photos as the ONLY visual source of truth.

=== CRITICAL REMINDER ===
1. RULE #1 (MOST IMPORTANT): Make every product EXACTLY the same as in the source
   document/images — identical design, colors, markings, shape, proportions, branding.
   If you are not 100% sure a product matches exactly, DO NOT put it in the image — SKIP IT.
* DO NOT CREATE any product, figure, character, variant, colorway, SKU, or packaging that is NOT in the attached source document/images.
* Every item in the final photo MUST be 100% consistent with the given products shown in the source files.
* Do NOT invent extra series members or filler items. When in doubt, show FEWER products from the source.
* Variation between images may change camera angle, lighting, or staging ONLY — never add or swap products.

Target Selection:
Use ONLY the exact selected products listed in ALLOWED_PRODUCTS below. Every product shown in the final photo MUST be visually matchable to an item that appears in the attached source image(s).

Strict Rules:
1. RULE #1: Products must be EXACT copies of the source. If unsure — skip that product entirely. Do not approximate.
* Do NOT create, invent, hallucinate, or freestyle any product not clearly present in the attached source document/images.
* If ALLOWED_PRODUCTS contains a series title, only depict the specific figures/items that are visibly shown in the attached source images for that series.
* Product appearance MUST be 100% consistent with the source imagery. Copy the real product look — do NOT stylize, redesign, or approximate.
* Do NOT add "similar looking" products that are not in the document.
* Scale & proportions MUST be physically realistic:
  * If a product has packaging, the product itself must NEVER appear larger than its packaging.
  * Keep relative scale correct across items in the same scene.

Objective:
Take the ALLOWED_PRODUCTS that are visible in the uploaded source files and generate a professional, high-resolution social media image (optimized for Instagram and Facebook) showcasing a harmonious, multi-item Japanese-style retail collection display.

Aesthetic & Style Requirements:
* Style: Professional Japanese minimalist commercial photography, resembling a high-end Tokyo boutique shelf display or traditional Wabi-Sabi gallery setting.
* Setting & Composition:
  * Arrange the allowed products together in a balanced, curated flat-lay or tiered staging setup.
  * Natural textures: Warm light-toned wood (Hinoki or Ash), smooth river stones, textured linen fabrics, subtle bamboo elements, or matte ceramic pedestals.
  * Background elements: Softly blurred shoji screen textures, a gentle ray of natural morning sunlight filtering through a window, or tasteful minimal floral accents.
* Lighting: Soft, diffused natural daylight with gentle, realistic shadows.
* Color Palette: Warm neutrals, muted earth tones, soft beige, light wood, and clean white accents.

Instructions:
1. RULE #1: Only include products you can make EXACTLY the same as the source. If not sure, skip them — do not put them in the image.
2. Inspect the attached source image(s) and identify ONLY the allowed products that actually appear there.
3. Compose a single cohesive Japanese-style commercial display using those exact products only (unmodified).
4. Before finalizing, verify every depicted product is in ALLOWED_PRODUCTS and is an exact visual match to the source.
5. Output a pristine final social media photo that looks like a real commercial camera shot of the collection.

=== FINAL REMINDER ===
RULE #1: Make the product EXACTLY the same as in the source. If not sure, do not put it — skip it.
DO NOT CREATE any product not in the document. MUST be 100% consistent with the given products in the attached source images.
"""


def fill_product_category(style_template: str, products: list[str]) -> str:
    """Replace ``[PRODUCT_CATEGORY]`` with the selected campaign products."""
    category = ", ".join(name.strip() for name in products if name.strip()) or "selected products"
    return style_template.replace("[PRODUCT_CATEGORY]", category)


def build_allowed_products_block(products: list[str]) -> str:
    """Build an explicit allowlist block for image generation."""
    lines = [
        "ALLOWED_PRODUCTS (exact allowlist — show ONLY these, and only if visible in source images):",
    ]
    cleaned = [name.strip() for name in products if name and name.strip()]
    if not cleaned:
        lines.append("- (none provided)")
    else:
        for idx, name in enumerate(cleaned, start=1):
            lines.append(f"{idx}. {name}")
    lines.append(
        "If a listed name is a series/category, include only items clearly visible "
        "in the attached source images. NEVER invent additional members."
    )
    return "\n".join(lines)


def is_blank_negative_prompt(value: str | None) -> bool:
    """Return True when the negative prompt should be treated as unused."""
    if value is None:
        return True
    normalized = value.strip()
    return normalized in {"", ".", "none", "None", "NONE", "n/a", "N/A"}


def build_negative_prompt_block(negative_prompt: str | None) -> str:
    """Build a negative-prompt instruction block, or empty string if unused."""
    if is_blank_negative_prompt(negative_prompt):
        return ""
    cleaned = (negative_prompt or "").strip()
    return (
        "NEGATIVE PROMPT (hard exclusions):\n"
        f"{cleaned}\n"
        "Skip / omit / do NOT include ANY of the elements mentioned above. "
        "If the style brief conflicts with the negative prompt, the negative prompt wins."
    )


def compose_image_generation_prompt(
    *,
    style_template: str,
    products: list[str],
    negative_prompt: str | None = None,
    variation_note: str = "",
) -> str:
    """Compose the final image-generation prompt with allowlist + negative rules."""
    filled = fill_product_category(style_template, products)
    parts = [
        filled.rstrip(),
        "",
        build_allowed_products_block(products),
    ]
    negative_block = build_negative_prompt_block(negative_prompt)
    if negative_block:
        parts.extend(["", negative_block])
    if variation_note.strip():
        parts.extend(["", variation_note.strip()])
    parts.extend(
        [
            "",
            "=== FINAL CHECK ===",
            "1. RULE #1: Every product must be EXACTLY the same as in the source. "
            "If not sure, do NOT put it — SKIP it.",
            "2. Every product/figure must appear in the attached source document/images.",
            "3. Every product must be in ALLOWED_PRODUCTS and visually match the source.",
            "4. Do NOT invent products. Prefer fewer exact products over approximate ones.",
            "5. Variation is composition/lighting ONLY — never add new products.",
        ]
    )
    return "\n".join(parts)
