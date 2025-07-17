#!/usr/bin/env python3
"""
multi_lib_pdf_layout_inspect.py

Compare how different Python PDF libraries "see" a PDF page.

Supported (if installed):
    • PyMuPDF (fitz)
    • pdfplumber
    • PyPDF2  (limited positional data; shows page bbox only)

For each available library:
    • Extract per-page text (and bounding boxes when supported)
    • Extract embedded images (when supported)
    • Render a PNG visualization with bounding boxes + short text labels
    • Dump per-page JSON metadata
    • Save extracted image binaries

Edit CONFIG below and run this file directly.
"""

import os
import io
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PDF_PATH = "sample.pdf"                # <--- change me
OUTDIR = "multi_lib_layout_output"     # root outdir
INVERT_Y = True                        # flip Y so top of page is visually at top
MAX_LABEL_CHARS = 40                   # preview text length
DPI = 150                              # PNG output resolution

# Per-lib toggles
RUN_PYMUPDF = True
RUN_PDFPLUMBER = True
RUN_PYPDF2 = True

# Extract images?
EXTRACT_IMAGES = True


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multi_lib_pdf_layout")


# ------------------------------------------------------------------
# Common data classes
# ------------------------------------------------------------------
@dataclass
class BoxInfo:
    """Generic bounding box + text."""
    page: int
    idx: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

@dataclass
class ImgBoxInfo:
    """Generic image bbox + saved path (if extracted)."""
    page: int
    idx: int
    x0: float
    y0: float
    x1: float
    y1: float
    saved_path: Optional[str]


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_label(text: str, max_chars: int = MAX_LABEL_CHARS) -> str:
    t = " ".join(text.split())
    if len(t) > max_chars:
        t = t[: max_chars - 1] + "…"
    return t


def add_rect(ax, bbox: Tuple[float, float, float, float], label=None, linewidth=1.0, linestyle="-"):
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    rect = plt.Rectangle(
        (x0, y0),
        w,
        h,
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


# ==================================================================
# PyMuPDF (fitz) SECTION
# ==================================================================
def pymupdf_extract(pdf_path: str):
    """
    Use PyMuPDF to extract text blocks & images w/ coords.
    Returns list of dict per page: {"page_num", "width", "height", "text_boxes": [...], "images": [...]}
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping.")
        return None

    doc = fitz.open(pdf_path)
    pages_data = []

    for pnum in range(len(doc)):
        page = doc[pnum]
        page_w, page_h = page.rect.width, page.rect.height

        # TEXT: get text blocks (dict mode)
        # Each block: [x0, y0, x1, y1, "text", block_no, block_type]
        # We'll use page.get_text("blocks") which returns a list of tuples
        blocks_raw = page.get_text("blocks")  # list of tuples
        text_boxes: List[BoxInfo] = []
        tb_idx = 0
        for blk in blocks_raw:
            x0, y0, x1, y1, text, block_no = blk[:6]
            # Sometimes text comes w/ trailing newline
            if text.strip():
                text_boxes.append(
                    BoxInfo(
                        page=pnum + 1,
                        idx=tb_idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text=text,
                    )
                )
                tb_idx += 1

        # IMAGES
        images: List[ImgBoxInfo] = []
        if EXTRACT_IMAGES:
            im_outdir = os.path.join(OUTDIR, "pymupdf", f"page_{pnum+1:03d}_images")
            ensure_dir(im_outdir)

            im_idx = 0
            for img in page.get_images(full=True):
                xref = img[0]
                # Extract pixmap
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Some images are CMYK etc; convert if needed
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_path = os.path.join(im_outdir, f"p{pnum+1:03d}_img{im_idx:03d}.png")
                    pix.save(img_path)
                except Exception as e:
                    logger.warning(f"PyMuPDF failed to export image on page {pnum+1}, idx {im_idx}: {e}")
                    img_path = None

                # Approximate bbox: locate image rects through page.get_image_info?
                # Simpler: search for image rects via page.get_text("rawdict") which includes spans/images.
                # We'll quickly query all image bbox from rawdict objects.
                img_rect = None
                # fallback: None; we'll fill after pass
                images.append(
                    ImgBoxInfo(
                        page=pnum + 1,
                        idx=im_idx,
                        x0=0.0,
                        y0=0.0,
                        x1=0.0,
                        y1=0.0,
                        saved_path=img_path,
                    )
                )
                im_idx += 1

            # Fill in bbox by scanning rawdict (contains "image" blocks)
            rawdict = page.get_text("rawdict")
            # rawdict["blocks"] list; each block has type 1=image
            for blk in rawdict.get("blocks", []):
                if blk["type"] == 1:
                    # bounding box
                    x0, y0, x1, y1 = blk["bbox"]
                    # assign to next available placeholder ImgBoxInfo w/ 0 sized box
                    for im in images:
                        if im.x0 == im.x1 == 0.0 and im.y0 == im.y1 == 0.0:
                            im.x0, im.y0, im.x1, im.y1 = x0, y0, x1, y1
                            break

        else:
            images = []

        pages_data.append(
            {
                "page_num": pnum + 1,
                "width": page_w,
                "height": page_h,
                "text_boxes": text_boxes,
                "images": images,
            }
        )

    doc.close()
    return pages_data


def pymupdf_visualize_page(page_dict, invert_y=INVERT_Y, out_png=None):
    fig, ax = plt.subplots(figsize=(8, 10))
    w = page_dict["width"]
    h = page_dict["height"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    add_rect(ax, (0, 0, w, h), label="PAGE", linewidth=1.5, linestyle="--")

    for tb in page_dict["text_boxes"]:
        lbl = f"T{tb.idx}: {sanitize_label(tb.text)}"
        add_rect(ax, (tb.x0, tb.y0, tb.x1, tb.y1), label=lbl, linewidth=0.8)

    for im in page_dict["images"]:
        lbl = f"I{im.idx}"
        add_rect(ax, (im.x0, im.y0, im.x1, im.y1), label=lbl, linewidth=1.0, linestyle="-.")

    ax.set_title(f"PyMuPDF Page {page_dict['page_num']}")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


# ==================================================================
# pdfplumber SECTION
# ==================================================================
def pdfplumber_extract(pdf_path: str):
    """
    Use pdfplumber to extract char-level text & image metadata.
    We aggregate chars into word-like boxes (use built-in extract_words()).
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; skipping.")
        return None

    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages, start=1):
            w, h = page.width, page.height

            # Words
            words = page.extract_words()  # list of dict: x0, x1, top, bottom, text
            text_boxes: List[BoxInfo] = []
            tb_idx = 0
            for wdict in words:
                text = wdict.get("text", "").strip()
                if not text:
                    continue
                x0 = float(wdict["x0"])
                x1 = float(wdict["x1"])
                # pdfplumber uses top/bottom (y origin top)
                top = float(wdict["top"])
                bottom = float(wdict["bottom"])
                # Convert to pdf coordinate w/ origin bottom-left: y0=page_h-bottom, y1=page_h-top
                y0 = h - bottom
                y1 = h - top
                text_boxes.append(
                    BoxInfo(
                        page=pnum,
                        idx=tb_idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text=text,
                    )
                )
                tb_idx += 1

            # Images
            images: List[ImgBoxInfo] = []
            im_idx = 0
            if EXTRACT_IMAGES:
                im_outdir = os.path.join(OUTDIR, "pdfplumber", f"page_{pnum:03d}_images")
                ensure_dir(im_outdir)

            for img in page.images:  # list of dicts
                # pdfplumber image coords: x0,y0,x1,y1 w/ origin top-left? Actually x0,top,width,height
                # We'll reconstruct properly:
                x0 = float(img["x0"])
                # top is distance from top
                top = float(img["top"])
                width = float(img["width"])
                height = float(img["height"])
                x1 = x0 + width
                # pdfplumber coordinate transform
                y1 = h - top
                y0 = y1 - height

                saved_path = None
                if EXTRACT_IMAGES:
                    try:
                        im = page.crop((x0, top, x1, top + height)).to_image(resolution=150)
                        saved_path = os.path.join(im_outdir, f"p{pnum:03d}_img{im_idx:03d}.png")
                        im.save(saved_path, format="PNG")
                    except Exception as e:
                        logger.warning(f"pdfplumber failed to save image p{pnum} idx {im_idx}: {e}")
                        saved_path = None

                images.append(
                    ImgBoxInfo(
                        page=pnum,
                        idx=im_idx,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        saved_path=saved_path,
                    )
                )
                im_idx += 1

            pages_data.append(
                {
                    "page_num": pnum,
                    "width": w,
                    "height": h,
                    "text_boxes": text_boxes,
                    "images": images,
                }
            )

    return pages_data


def pdfplumber_visualize_page(page_dict, invert_y=INVERT_Y, out_png=None):
    fig, ax = plt.subplots(figsize=(8, 10))
    w = page_dict["width"]
    h = page_dict["height"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    add_rect(ax, (0, 0, w, h), label="PAGE", linewidth=1.5, linestyle="--")

    for tb in page_dict["text_boxes"]:
        lbl = f"T{tb.idx}: {sanitize_label(tb.text)}"
        add_rect(ax, (tb.x0, tb.y0, tb.x1, tb.y1), label=lbl, linewidth=0.5)

    for im in page_dict["images"]:
        lbl = f"I{im.idx}"
        add_rect(ax, (im.x0, im.y0, im.x1, im.y1), label=lbl, linewidth=1.0, linestyle="-.")

    ax.set_title(f"pdfplumber Page {page_dict['page_num']}")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


# ==================================================================
# PyPDF2 SECTION
# ==================================================================
def pypdf2_extract(pdf_path: str):
    """
    PyPDF2 exposes text (no layout grouping) and page mediabox coords.
    We'll produce a single box per page w/ full extracted text.
    """
    try:
        import PyPDF2
    except ImportError:
        logger.warning("PyPDF2 not installed; skipping.")
        return None

    pages_data = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for pnum, page in enumerate(reader.pages, start=1):
            # mediabox is [llx, lly, urx, ury]
            mediabox = page.mediabox
            x0 = float(mediabox.left)
            y0 = float(mediabox.bottom)
            x1 = float(mediabox.right)
            y1 = float(mediabox.top)
            w = x1 - x0
            h = y1 - y0

            text = ""
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"PyPDF2 extract_text failed page {pnum}: {e}")

            text_boxes = [
                BoxInfo(
                    page=pnum,
                    idx=0,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    text=text,
                )
            ]

            # PyPDF2 image extraction is non-trivial; skip (metadata only).
            images: List[ImgBoxInfo] = []

            pages_data.append(
                {
                    "page_num": pnum,
                    "width": w,
                    "height": h,
                    "text_boxes": text_boxes,
                    "images": images,
                }
            )

    return pages_data


def pypdf2_visualize_page(page_dict, invert_y=INVERT_Y, out_png=None):
    fig, ax = plt.subplots(figsize=(8, 10))
    w = page_dict["width"]
    h = page_dict["height"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    add_rect(ax, (0, 0, w, h), label="PAGE", linewidth=1.5, linestyle="--")

    # Single full-page text box
    tb = page_dict["text_boxes"][0]
    lbl = f"T0: {sanitize_label(tb.text)}"
    add_rect(ax, (tb.x0, tb.y0, tb.x1, tb.y1), label=lbl, linewidth=0.8)

    ax.set_title(f"PyPDF2 Page {page_dict['page_num']}")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


# ==================================================================
# JSON save helper (common)
# ==================================================================
def save_page_json(page_dict, out_json_path: str):
    data = {
        "page_num": page_dict["page_num"],
        "width": page_dict["width"],
        "height": page_dict["height"],
        "text_boxes": [asdict(tb) for tb in page_dict["text_boxes"]],
        "images": [asdict(im) for im in page_dict["images"]],
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {out_json_path}")


# ==================================================================
# MAIN
# ==================================================================
def main():
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    ensure_dir(OUTDIR)

    # ---------------------- PyMuPDF ----------------------
    if RUN_PYMUPDF:
        _data = pymupdf_extract(PDF_PATH)
        if _data is not None:
            lib_dir = os.path.join(OUTDIR, "pymupdf")
            ensure_dir(lib_dir)
            for page_dict in _data:
                pnum = page_dict["page_num"]
                png_path = os.path.join(lib_dir, f"pymupdf_page_{pnum:03d}.png")
                pymupdf_visualize_page(page_dict, invert_y=INVERT_Y, out_png=png_path)
                json_path = os.path.join(lib_dir, f"pymupdf_page_{pnum:03d}.json")
                save_page_json(page_dict, json_path)

    # ---------------------- pdfplumber -------------------
    if RUN_PDFPLUMBER:
        _data = pdfplumber_extract(PDF_PATH)
        if _data is not None:
            lib_dir = os.path.join(OUTDIR, "pdfplumber")
            ensure_dir(lib_dir)
            for page_dict in _data:
                pnum = page_dict["page_num"]
                png_path = os.path.join(lib_dir, f"pdfplumber_page_{pnum:03d}.png")
                pdfplumber_visualize_page(page_dict, invert_y=INVERT_Y, out_png=png_path)
                json_path = os.path.join(lib_dir, f"pdfplumber_page_{pnum:03d}.json")
                save_page_json(page_dict, json_path)

    # ---------------------- PyPDF2 -----------------------
    if RUN_PYPDF2:
        _data = pypdf2_extract(PDF_PATH)
        if _data is not None:
            lib_dir = os.path.join(OUTDIR, "pypdf2")
            ensure_dir(lib_dir)
            for page_dict in _data:
                pnum = page_dict["page_num"]
                png_path = os.path.join(lib_dir, f"pypdf2_page_{pnum:03d}.png")
                pypdf2_visualize_page(page_dict, invert_y=INVERT_Y, out_png=png_path)
                json_path = os.path.join(lib_dir, f"pypdf2_page_{pnum:03d}.json")
                save_page_json(page_dict, json_path)

    logger.info("All done.")


if __name__ == "__main__":
    main()
