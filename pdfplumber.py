#!/usr/bin/env python3
"""
pdfminer_layout_inspect.py

Visual debug utility: see how pdfminer.six interprets each page of a PDF.

What it does:
    • Parses PDF into LTPage layout objects.
    • Extracts LTTextBox text, coords.
    • Extracts LTImage raw bytes (best effort) & saves to files.
    • Draws bounding boxes over page & annotates each text box with short text preview.
    • Saves per-page PNG visualizations + JSON metadata.

Edit the CONFIG section below (PDF_PATH, OUTDIR) and run:
    python pdfminer_layout_inspect.py
"""

import os
import io
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import matplotlib.pyplot as plt

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTPage,
    LTTextBox,
    LTTextLine,
    LTImage,
    LTFigure,
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PDF_PATH = "sample.pdf"          # <--- change to your PDF file
OUTDIR   = "layout_debug_output" # All outputs written here
SHOW_TEXT_LINES = False          # Draw LTTextLine boxes (can get noisy)
INVERT_Y = True                  # Flip Y so page looks upright in image
MAX_LABEL_CHARS = 40             # Max chars to show in on-page labels
SAVE_JSON = True                 # Dump per-page JSON
EXTRACT_IMAGES = True            # Attempt to save embedded images
DPI = 150                        # Output PNG resolution


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pdfminer_layout_inspect")


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
@dataclass
class TextBoxInfo:
    page: int
    idx: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

@dataclass
class ImageInfo:
    page: int
    idx: int
    x0: float
    y0: float
    x1: float
    y1: float
    saved_path: Optional[str]  # None if not saved / unknown


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_label(text: str, max_chars: int = 40) -> str:
    """Strip and compress whitespace; truncate to max_chars for display."""
    t = " ".join(text.split())
    if len(t) > max_chars:
        t = t[:max_chars - 1] + "…"
    return t


def add_rect(ax, bbox, label=None, linewidth=1.0, linestyle="-"):
    """
    Draw rectangle patch.
    bbox: (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    rect = plt.Rectangle(
        (x0, y0), w, h,
        fill=False,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x0,
            y1,
            label,
            fontsize=6,
            va="bottom",
            ha="left",
        )


def iter_layout(layout_obj):
    """Depth-first iterator over layout tree."""
    yield layout_obj
    if hasattr(layout_obj, "__iter__"):
        for child in layout_obj:
            yield from iter_layout(child)


# --------------- Image Extraction ------------------
def guess_image_ext(stream_bytes: bytes) -> str:
    """Heuristic: return file extension based on magic bytes."""
    if stream_bytes.startswith(b"\xff\xd8"):
        return ".jpg"
    if stream_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if stream_bytes.startswith(b"GIF87a") or stream_bytes.startswith(b"GIF89a"):
        return ".gif"
    if stream_bytes.startswith(b"BM"):
        return ".bmp"
    if stream_bytes[0:4] == b"RIFF" and stream_bytes[8:12] == b"WEBP":
        return ".webp"
    # Fallback
    return ".bin"


def save_ltimage(lt_image: LTImage, out_base: str) -> Optional[str]:
    """
    Attempt to extract raw image bytes from LTImage and save.
    Returns saved path or None if failed.
    """
    if not hasattr(lt_image, "stream") or lt_image.stream is None:
        return None

    try:
        raw_bytes = lt_image.stream.get_rawdata()
    except Exception:
        # Some filters require get_data()
        try:
            raw_bytes = lt_image.stream.get_data()
        except Exception as e:
            logger.warning(f"Failed to get image data: {e}")
            return None

    ext = guess_image_ext(raw_bytes)
    out_path = out_base + ext
    try:
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
        return out_path
    except Exception as e:
        logger.warning(f"Failed to write image file: {e}")
        return None


# ------------------------------------------------------------------
# Visualization of a single page
# ------------------------------------------------------------------
def visualize_page(
    page_num: int,
    layout: LTPage,
    text_boxes: List[TextBoxInfo],
    images: List[ImageInfo],
    show_text_lines: bool = False,
    invert_y: bool = True,
    out_png: Optional[str] = None,
):
    """Render a page with bounding boxes & labels."""
    fig, ax = plt.subplots(figsize=(8, 10))

    # Basic axes config
    ax.set_xlim(0, layout.width)
    ax.set_ylim(0, layout.height)
    ax.set_aspect("equal", adjustable="box")

    if invert_y:
        ax.invert_yaxis()  # so top of page is visually at top

    # Page border
    add_rect(ax, (0, 0, layout.width, layout.height), label="PAGE", linewidth=1.5, linestyle="--")

    # Draw text boxes
    for tb in text_boxes:
        lbl = f"T{tb.idx}: {sanitize_label(tb.text, MAX_LABEL_CHARS)}"
        add_rect(ax, (tb.x0, tb.y0, tb.x1, tb.y1), label=lbl, linewidth=0.8)

    # Draw text lines (optional)
    if show_text_lines:
        for obj in iter_layout(layout):
            if isinstance(obj, LTTextLine):
                add_rect(ax, (obj.x0, obj.y0, obj.x1, obj.y1), label=None, linewidth=0.5, linestyle=":")

    # Draw images
    for im in images:
        lbl = f"I{im.idx}"
        if im.saved_path:
            lbl += " img"
        add_rect(ax, (im.x0, im.y0, im.x1, im.y1), label=lbl, linewidth=1.0, linestyle="-.")

    ax.set_title(f"Page {page_num} (w={layout.width:.0f}, h={layout.height:.0f})")
    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=DPI)
        logger.info(f"Saved page {page_num} visualization: {out_png}")
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
# Extraction pipeline
# ------------------------------------------------------------------
def extract_pdf(pdf_path: str):
    """
    Parse PDF and collect per-page text boxes & images.
    Returns list of dicts: {"page_num": int, "layout": LTPage, "text_boxes": [...], "images": [...]}
    """
    laparams = LAParams(
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        detect_vertical=True,
        all_texts=True,
    )

    pages_data = []
    for page_num, layout in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
        if not isinstance(layout, LTPage):
            logger.warning(f"Unexpected layout type for page {page_num}: {type(layout)}")

        text_boxes: List[TextBoxInfo] = []
        images: List[ImageInfo] = []

        # Collect objects
        tb_idx = 0
        im_idx = 0

        for obj in iter_layout(layout):
            if isinstance(obj, LTTextBox):
                text = obj.get_text() if hasattr(obj, "get_text") else ""
                text_boxes.append(
                    TextBoxInfo(
                        page=page_num,
                        idx=tb_idx,
                        x0=obj.x0,
                        y0=obj.y0,
                        x1=obj.x1,
                        y1=obj.y1,
                        text=text,
                    )
                )
                tb_idx += 1

            elif isinstance(obj, LTImage):
                # We'll save later once we know output directory
                images.append(
                    ImageInfo(
                        page=page_num,
                        idx=im_idx,
                        x0=obj.x0,
                        y0=obj.y0,
                        x1=obj.x1,
                        y1=obj.y1,
                        saved_path=None,
                    )
                )
                im_idx += 1

        pages_data.append(
            {
                "page_num": page_num,
                "layout": layout,
                "text_boxes": text_boxes,
                "images": images,
            }
        )

    return pages_data


# ------------------------------------------------------------------
# Save JSON metadata
# ------------------------------------------------------------------
def save_page_json(page_dict, out_json_path: str):
    """
    Save text boxes + images (coords + text + saved image path) to JSON.
    `page_dict` is one element from extract_pdf() results.
    """
    data = {
        "page_num": page_dict["page_num"],
        "width": page_dict["layout"].width,
        "height": page_dict["layout"].height,
        "text_boxes": [asdict(tb) for tb in page_dict["text_boxes"]],
        "images": [asdict(im) for im in page_dict["images"]],
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {out_json_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF_PATH not found: {PDF_PATH}")

    ensure_outdir(OUTDIR)

    logger.info(f"Parsing: {PDF_PATH}")
    pages_data = extract_pdf(PDF_PATH)

    # Extract & save image binaries
    if EXTRACT_IMAGES:
        # Re-parse pages w/ layout iteration to access LTImage objects again (since we didn't store refs)
        laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            detect_vertical=True,
            all_texts=True,
        )
        # We'll walk side by side: pages_data + fresh extract_pages generator
        fresh_pages = list(extract_pages(PDF_PATH, laparams=laparams))
        if len(fresh_pages) != len(pages_data):
            logger.warning("Page count mismatch during image extraction; continuing best-effort.")

        for page_dict in pages_data:
            pnum = page_dict["page_num"]
            layout = fresh_pages[pnum - 1] if pnum - 1 < len(fresh_pages) else None
            if layout is None:
                continue

            # Map images by index in traversal order
            im_outdir = os.path.join(OUTDIR, f"page_{pnum:03d}_images")
            ensure_outdir(im_outdir)

            im_idx = 0
            for obj in iter_layout(layout):
                if isinstance(obj, LTImage):
                    base = os.path.join(im_outdir, f"p{pnum:03d}_img{im_idx:03d}")
                    saved = save_ltimage(obj, base)
                    # record
                    for iminfo in page_dict["images"]:
                        if iminfo.idx == im_idx:
                            iminfo.saved_path = saved
                            break
                    im_idx += 1

    # Visualization & JSON dump
    for page_dict in pages_data:
        pnum = page_dict["page_num"]
        layout = page_dict["layout"]
        text_boxes = page_dict["text_boxes"]
        images = page_dict["images"]

        out_png = os.path.join(OUTDIR, f"page_{pnum:03d}.png")
        visualize_page(
            pnum,
            layout,
            text_boxes,
            images,
            show_text_lines=SHOW_TEXT_LINES,
            invert_y=INVERT_Y,
            out_png=out_png,
        )

        if SAVE_JSON:
            out_json = os.path.join(OUTDIR, f"page_{pnum:03d}.json")
            save_page_json(page_dict, out_json)

    logger.info("Done.")


if __name__ == "__main__":
    main()
