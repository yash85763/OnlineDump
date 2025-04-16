import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
from pdf2image import convert_from_path
import json
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
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
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model(model_path):
    """Load the InternVL model and tokenizer."""
    print(f"Loading model from {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    """Convert PDF pages to images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Converting PDF {pdf_path} to images...")
    images = convert_from_path(pdf_path, dpi=dpi)
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'page_{i+1}.jpg')
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
    
    print(f"Converted {len(image_paths)} pages to images")
    return image_paths

def process_pdf_with_vlm(model, tokenizer, pdf_path, output_dir, save_dir, max_num=12):
    """Process a PDF file with the VLM model."""
    # Create image directory
    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, image_dir)
    
    # Process each page with the VLM
    results = []
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing pages")):
        page_num = i + 1
        print(f"Processing page {page_num}/{len(image_paths)}")
        
        # Load and prepare the image
        pixel_values = load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        
        # Generate prompts for different aspects of the document
        prompts = [
            "<image>\nExtract the hierarchical structure of this document page, identifying all headings, subheadings, and their content. Format the output as structured text maintaining the hierarchy.",
            "<image>\nIdentify and extract any tables, lists, or structured elements in this document page.",
            "<image>\nExtract any figures, charts, or visual elements on this page and describe their content."
        ]
        
        page_results = {}
        for j, prompt in enumerate(prompts):
            print(f"Running prompt {j+1}/{len(prompts)}")
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            
            if j == 0:
                page_results["structure"] = response
            elif j == 1:
                page_results["tables_lists"] = response
            elif j == 2:
                page_results["figures"] = response
        
        page_results["page_number"] = page_num
        results.append(page_results)
    
    # Save results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save as JSON
    json_path = os.path.join(save_dir, "pdf_parsed_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as text
    text_path = os.path.join(save_dir, "pdf_parsed_results.txt")
    with open(text_path, 'w') as f:
        for page in results:
            f.write(f"PAGE {page['page_number']}\n")
            f.write("="*80 + "\n\n")
            f.write("DOCUMENT STRUCTURE:\n")
            f.write(page['structure'] + "\n\n")
            f.write("TABLES AND LISTS:\n")
            f.write(page['tables_lists'] + "\n\n")
            f.write("FIGURES AND VISUALS:\n")
            f.write(page['figures'] + "\n\n")
            f.write("-"*80 + "\n\n")
    
    print(f"Results saved to {save_dir}")
    return results

def save_model_local(model, tokenizer, save_dir):
    """Save the fine-tuned model to a local directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Saving model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def main():
    # Hardcoded parameters
    pdf_path = "path/to/your/document.pdf"  # CHANGE THIS TO YOUR PDF PATH
    model_path = "OpenGVLab/InternVL2-8B"
    output_dir = "./output"
    save_dir = "./results"
    model_save_dir = "./saved_model"
    max_num = 12
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Process PDF
    process_pdf_with_vlm(
        model=model,
        tokenizer=tokenizer,
        pdf_path=pdf_path,
        output_dir=output_dir,
        save_dir=save_dir,
        max_num=max_num
    )
    
    # Save model
    save_model_local(model, tokenizer, model_save_dir)

if __name__ == "__main__":
    main()
