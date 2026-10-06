# Post Maker

CLI tool that turns product catalogs into a ready-to-post social media campaign.

Drop PDF / DOCX / image / text files into `inputs/`, select pages and products, then generate:

- Collection-style campaign images (Vertex AI)
- Platform posts for Instagram, Facebook, and X
- Token usage report per run

## Features

- Reads catalogs from `inputs/` (PDF, DOCX, TXT/MD, JPG/PNG/WEBP)
- PDF page selection (`1,2,5` or `1-3`)
- Product detection via text patterns + Vertex vision (works on scanned PDFs)
- Product selection by number, range, name, or `all` (e.g. `1-7,9`)
- One campaign folder with multi-product images + platform posts
- Document fidelity rule: products must match the source; if unsure, skip
- AI masters generated once, then locally resized per platform (no extra image tokens)
- Token usage tracked and saved to `token_usage.txt`

## Requirements

- Python 3.11+
- Google Cloud project with Vertex AI enabled
- Service account credentials (`GOOGLE_APPLICATION_CREDENTIALS`)

## Setup

```bash
cd post_maker
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional model overrides
BU_GEMINI_MODEL=gemini-2.5-flash-lite
BU_IMAGE_MODEL=gemini-3.1-flash-lite-image
DEFAULT_IMAGE_COUNT=2
```

## Quick start

1. Put catalog files in `inputs/` (or a subfolder).
2. Run:

```bash
python main.py
```

3. Follow the prompts:
   - PDF pages to read
   - Products to include
   - Content direction
   - Image count (default: 2)
   - Image style / negative prompt

4. Find results under `outputs/{campaign}/`.

### Non-interactive example

```bash
python main.py \
  --pages 1 \
  --items "1-3,5" \
  --direction "Introduce these product(s) or series" \
  --image-count 2 \
  --platforms instagram,facebook,x
```

Parse and select only (no generation):

```bash
python main.py --pages 1 --items "1-7,9" --skip-generate
```

## CLI options

| Flag | Description |
|------|-------------|
| `--input-subdir` | Subfolder under `inputs/` (e.g. `catalog_2026`) |
| `--pages` | PDF pages, e.g. `1,2,5` or `1-3` |
| `--items` | Product names, numbers, or ranges, e.g. `1-7,9` |
| `--all-items` | Use every detected product (no prompt) |
| `--direction` | Content direction for posts |
| `--image-count` | Number of AI master images (default: 2) |
| `--image-style` | Custom image style prompt |
| `--image-style-file` | Path to a style prompt text file |
| `--negative-prompt` | Elements to exclude from images (`.` = none) |
| `--word-count` | Approx. post length (default: 50) |
| `--platforms` | `instagram,facebook,x` (default: all) |
| `--ocr` | Local OCR for image inputs if pytesseract is available |
| `--skip-generate` | Parse / detect / search only |
| `-v` / `--verbose` | Debug logging |

## Workflow

```text
inputs/  →  parse docs  →  select PDF pages  →  vision enrich
       →  detect products  →  select items  →  configure campaign
       →  generate images + posts  →  outputs/{campaign}/
```

1. **Scan** files under `inputs/`
2. **Select PDF pages** (interactive or `--pages`)
3. **Enrich** scanned/catalog pages with Vertex vision
4. **Detect & select products** (menu, ranges, names, or `--all-items`)
5. **Configure** direction, image count, style, negative prompt, platforms
6. **Generate** N AI master images + platform posts
7. **Resize** masters locally into each platform folder

## Project layout

```text
post_maker/
├── main.py                 # CLI entry
├── requirements.txt
├── .env                    # credentials & model settings (create locally)
├── inputs/                 # catalog source files
├── outputs/                # generated campaigns
├── docs/                   # system document PDF + generator
├── skills/
│   ├── information_skill.py   # detect / search products
│   └── social_media_skill.py  # images + posts
└── src/post_maker/
    ├── config.py
    ├── document_loader.py
    ├── image_style.py
    ├── models.py
    └── vertex_client.py
```

## Output layout

```text
outputs/{campaign}/
├── products.txt
├── image_style.txt
├── negative_prompt.txt
├── campaign_overview.txt
├── token_usage.txt
├── _source/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── prompt_image_N.txt
├── instagram/
│   ├── post.txt
│   └── image_N.jpg
├── facebook/
│   └── ...
└── x/
    └── ...
```

AI is called once per master image. Instagram / Facebook / X variants are Pillow resizes (0 extra image tokens).

## Document fidelity

Image prompts enforce:

1. Products must look **exactly** like the source catalog images
2. If unsure, **skip** the product — do not invent or approximate
3. No extra series members or filler SKUs

Source catalog page images are attached as reference for generation.

## Docs

- One-page overview: [`docs/post_maker_system_document.pdf`](docs/post_maker_system_document.pdf)
- Regenerate: `python docs/generate_system_document.py`

## Notes

- Default text model: `gemini-2.5-flash-lite`
- Default image model: `gemini-3.1-flash-lite-image` (Nano Banana 2 Lite)
- Keep secrets in `.env`; do not commit credentials
