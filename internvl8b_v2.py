import fitz  # PyMuPDF
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# --------------------------------------------------------------------------
# (1) Existing internVL2 transformations & functions
# --------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Additional tie-breaking based on area (optional)
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Splits an image into tiles based on the 'closest aspect ratio' approach.
    If use_thumbnail=True and the image was split, also adds a single thumbnail tile.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Build candidate aspect ratios (i,j) for i*j in [min_num, max_num].
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Resize the image to fit that aspect ratio
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        # Add a single thumbnail tile
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    Load a SINGLE image file (e.g. PNG/JPG) from disk,
    apply dynamic_preprocess, then return a stacked tensor.
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)

    images = dynamic_preprocess(
        image, image_size=input_size,
        use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(im) for im in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --------------------------------------------------------------------------
# (2) New PDF-specific logic
# --------------------------------------------------------------------------

def load_pdf_pages(pdf_path, resolution=200, max_pages=None):
    """
    Convert each page of the PDF into a high-resolution PIL image.
    Args:
      pdf_path    (str): Path to the input PDF.
      resolution (int):  DPI for rendering PDF pages (the higher the clearer, but also bigger memory).
      max_pages   (int): If set, limit the number of pages to load.
    Returns:
      A list of PIL.Images, one per page.
    """
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        # Render page at the given resolution
        mat = fitz.Matrix(resolution / 72, resolution / 72)  # 72 dpi is base
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # Convert to PIL
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(pil_img)
    return images

def parse_pdf_pages_with_model(
    pdf_path,
    model,
    tokenizer,
    generation_config,
    resolution=200,
    input_size=448,
    max_num=12,
    max_pages=None
):
    """
    (1) Convert each PDF page -> PIL image.
    (2) Process images into model-ready tensors.
    (3) For each page, run a single question to ask the model
        to parse the page into a hierarchical JSON structure.
    (4) Collect the responses in a dictionary.

    Returns:
      A Python dict with structure: {
         "pages": [
             { "page_index": int, "response": str (JSON text) },
             ...
         ]
      }
    """
    # 1) Convert each PDF page to a high-res image
    pil_pages = load_pdf_pages(pdf_path, resolution=resolution, max_pages=max_pages)

    # 2) For each page, get pixel_values
    parsed_results = {"pages": []}
    for page_index, pil_img in enumerate(pil_pages):
        # Build the tiling
        # (We reuse `dynamic_preprocess` + transform from the existing code)
        #   - You can also directly call `load_image` on a temporary file if you prefer.
        #   - Or keep it in-memory as done here:

        processed_imgs = dynamic_preprocess(
            pil_img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        transform = build_transform(input_size)
        tile_tensors = [transform(im) for im in processed_imgs]
        pixel_values = torch.stack(tile_tensors).to(torch.bfloat16).cuda()

        # 3) Ask the model to parse the page in JSON
        #    The prompt can be anything you like; here's an example:
        question = (
            "<image>\n"
            "Please parse the content of this PDF page and return a JSON structure "
            "reflecting its hierarchical organization (headings, subheadings, paragraphs, etc.). "
            "Only return valid JSON."
        )

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config
        )

        # 4) Collect the response
        parsed_results["pages"].append({
            "page_index": page_index + 1,
            "response": response
        })

    return parsed_results

# --------------------------------------------------------------------------
# (3) Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Load your model
    path = 'OpenGVLab/InternVL2-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    # 2) Set up generation configs
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=True
    )

    # 3) PDF you want to parse
    pdf_path = "my_document.pdf"

    # 4) Parse the PDF
    results = parse_pdf_pages_with_model(
        pdf_path, model, tokenizer, generation_config,
        resolution=200,   # Increase if needed for sharper text
        input_size=448,   # Image tile size
        max_num=12,       # Max number of tiles to split the page into
        max_pages=None    # or a specific integer limit
    )

    # 5) Print or save the results
    #    Each page's "response" string should ideally be JSON.
    #    However, the model might not always produce perfect JSON:
    #    you'll likely want to handle or sanitize the model outputs.
    print("=== PDF Parsing Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # If you want to save to a file:
    # with open("parsed_pdf.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
