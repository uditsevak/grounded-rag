"""Extract text from embedded PDF images via local OCR (Tesseract), so image
content becomes retrievable text alongside the surrounding page text.

Groq's vision-capable models (llama-3.2-*-vision-preview) were decommissioned
and no replacement is currently listed on the free tier (checked via the
live /models endpoint) — OCR is the dependency that's actually available and
free, and it directly fits diagrams/screenshots with rendered text. It won't
describe a photo with no text in it; that's a real limitation, not a bug.
"""
import io

import pytesseract
from PIL import Image


def caption_image(image_bytes: bytes, mime_type: str = "image/png") -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img).strip()
    except Exception as e:
        print(f"  [OCR failed: {e}; using placeholder]")
        return "[embedded image — OCR unavailable]"

    if not text:
        return "[embedded image — no text detected by OCR]"
    return " ".join(text.split())
