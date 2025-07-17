#!/usr/bin/env python3
"""
visualize_pdfminer_layout.py

Quick visualization of how pdfminer.six "sees" each PDF page.

For every page:
- Draw page boundary
- Draw rectangles around LTTextBox objects
- Draw rectangles around LTTextLine objects (optional flag)
- Draw rectangles around LTImage objects
- Draw rectangles around LTFigure objects (containers)

Outputs PNG files (page_001.png, etc.) and/or displays them interactively.

Example:
    python visualize_pdfminer_layout.py sample.pdf --outdir layout_imgs
"""

import os
import argparse
import logging

import matplotlib.pyplot as plt

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTPage,
    LTTextBox,
    LTTextLine,
    LTImage,
    LTFigure,
    LAParams,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pdfminer_layout_viz")


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------
def _add_rect(ax, bbox, label=None, linewidth=1.0, linestyle="-"):
    """
    Add a rectangle patch to axes.

    bbox: (x0, y0, x1, y1)
    Coordinates are already in PDF coordinate space (origin bottom-left).
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    rect = plt.Rectangle(
        (x0, y0),
        width,
        height,
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
            verticalalignment="bottom",
        )


def _iter_layout_objects(layout_obj):
    """
    Depth-first yield of all layout children.
    pdfminer LTPage / LTFigure / LTTextBox etc. objects are iterable
    over their children (where applicable).
    """
    yield layout_obj
    if hasattr(layout_obj, "__iter__"):
        for child in layout_obj:
            yield from _iter_layout_objects(child)


# ------------------------------------------------------------------
# Visualization core
# ------------------------------------------------------------------
def visualize_page_layout(
    layout,
    show_text_lines=False,
    show_images=True,
    show_figures=True,
    show_page_bbox=True,
    title_prefix="",
    ax=None,
):
    """
    Draw bounding boxes for a pdfminer LTPage layout.

    Parameters
    ----------
    layout : LTPage
    show_text_lines : bool
        If True, draw LTTextLine boxes inside text boxes.
    show_images : bool
        Draw LTImage boxes.
    show_figures : bool
        Draw LTFigure container boxes.
    show_page_bbox : bool
        Draw page boundary.
    title_prefix : str
        Prefix added to axes title (e.g., "Page 1 - ").
    ax : matplotlib.axes.Axes or None
        If None, create a new figure/axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Page dims
    page_w = layout.width
    page_h = layout.height

    # Configure axes to PDF coordinate space
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # comment out to use origin bottom-left visually; see note below

    # NOTE: Matplotlib's default origin is lower-left for data coords, but we
    # often invert_yaxis() to display page "as read" (top at top).
    # Remove ax.invert_yaxis() if you prefer PDF raw coordinates.

    # Page boundary
    if show_page_bbox:
        _add_rect(ax, (0, 0, page_w, page_h), label="PAGE", linewidth=1.5, linestyle="--")

    # Walk tree
    for obj in _iter_layout_objects(layout):
        if isinstance(obj, LTTextBox):
            _add_rect(ax, (obj.x0, obj.y0, obj.x1, obj.y1), label="TextBox", linewidth=0.8)
            if show_text_lines:
                for line in obj:
                    if isinstance(line, LTTextLine):
                        _add_rect(ax, (line.x0, line.y0, line.x1, line.y1), label="TextLn", linewidth=0.5, linestyle=":")
        elif show_images and isinstance(obj, LTImage):
            _add_rect(ax, (obj.x0, obj.y0, obj.x1, obj.y1), label="Image", linewidth=1.0, linestyle="-")
        elif show_figures and isinstance(obj, LTFigure):
            _add_rect(ax, (obj.x0, obj.y0, obj.x1, obj.y1), label="Figure", linewidth=1.0, linestyle="-.")
        # We skip other object types (curves, chars) for clarity.

    ax.set_title(f"{title_prefix}size=({page_w:.0f}x{page_h:.0f})")
    return ax


# ------------------------------------------------------------------
# Page extraction
# ------------------------------------------------------------------
def extract_pdf_layouts(pdf_path, laparams=None):
    """
    Generator yielding (page_number, LTPage) for each page in pdf_path.
    """
    if laparams is None:
        laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            detect_vertical=True,
            all_texts=True,
        )

    for page_num, layout in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
        if not isinstance(layout, LTPage):
            # extract_pages returns top-level LTPage objects, but just in case:
            logger.warning(f"Unexpected layout type on page {page_num}: {type(layout)}")
        yield page_num, layout


# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize pdfminer.six layout bounding boxes.")
    parser.add_argument("pdf", help="Path to PDF file.")
    parser.add_argument("--outdir", help="Directory to save page PNGs. If omitted, just display.", default=None)
    parser.add_argument("--no-show", action="store_true", help="Do not display windows (useful for batch).")
    parser.add_argument("--lines", action="store_true", help="Show LTTextLine boxes.")
    parser.add_argument("--no-images", action="store_true", help="Do not draw LTImage boxes.")
    parser.add_argument("--no-figures", action="store_true", help="Do not draw LTFigure boxes.")
    parser.add_argument("--keep-y-up", action="store_true", help="Do not invert y-axis (raw PDF coords).")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    laparams = LAParams(
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        detect_vertical=True,
        all_texts=True,
    )

    for page_num, layout in extract_pdf_layouts(pdf_path, laparams=laparams):
        fig, ax = plt.subplots(figsize=(8, 10))  # scale figure size as you like

        # Draw page
        visualize_page_layout(
            layout,
            show_text_lines=args.lines,
            show_images=not args.no_images,
            show_figures=not args.no_figures,
            show_page_bbox=True,
            title_prefix=f"Page {page_num} - ",
            ax=ax,
        )

        # Optionally keep raw PDF origin (y-up)
        if args.keep_y_up:
            ax.invert_yaxis()  # invert_yaxis() called in visualize; call again to undo
            ax.set_ylim(0, layout.height)  # ensure proper limits

        fig.tight_layout()

        if args.outdir:
            out_path = os.path.join(args.outdir, f"page_{page_num:03d}.png")
            fig.savefig(out_path, dpi=150)
            logger.info(f"Wrote {out_path}")

        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
