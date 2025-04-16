
# ! pip install poppler-utils
!apt-get install poppler-utils
! pip install pdf2image

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
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
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
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
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, dtype=torch.float32, device='cpu'):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to(dtype=dtype, device=device)
    return pixel_values

def load_model(model_path, use_gpu=True, device_map="auto", load_in_8bit=False, load_in_4bit=False):
    """Load the InternVL model and tokenizer with memory optimization options."""
    print(f"Loading model from {model_path}")
    
    # Create BitsAndBytesConfig if 4-bit or 8-bit quantization is requested
    quantization_config = None
    if load_in_4bit:
        print("Loading model in 4-bit quantization mode")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif load_in_8bit:
        print("Loading model in 8-bit quantization mode")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Memory optimization settings
    kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    # Set dtype based on device
    dtype = torch.bfloat16 if use_gpu and torch.cuda.is_available() else torch.float32
    kwargs["torch_dtype"] = dtype
    
    # Add device_map if specified
    if device_map is not None:
        kwargs["device_map"] = device_map
        
    # Add quantization config if specified
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    
    try:
        # Load model
        model = AutoModel.from_pretrained(model_path, **kwargs)
        
        # If not using device_map, explicitly move to appropriate device
        if device_map is None:
            if use_gpu and torch.cuda.is_available():
                print("Moving model to CUDA")
                model = model.cuda()
            else:
                print("Using CPU for inference (warning: this will be very slow)")
                model = model.cpu()
        
        model = model.eval()  # Set to evaluation mode
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("GPU out of memory error detected. Trying with more aggressive memory optimization...")
            if load_in_8bit:
                print("Falling back to 4-bit quantization")
                return load_model(model_path, use_gpu, device_map, False, True)
            elif not load_in_8bit and not load_in_4bit:
                print("Falling back to 8-bit quantization")
                return load_model(model_path, use_gpu, device_map, True, False)
            else:
                print("Falling back to CPU")
                return load_model(model_path, False, "cpu", False, False)
        else:
            raise e
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer, dtype

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

def process_pdf_with_vlm(model, tokenizer, pdf_path, output_dir, save_dir, max_num=12, use_gpu=True, batch_size=1, dtype=torch.float32):
    """Process a PDF file with the VLM model."""
    # Create image directory
    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, image_dir)
    
    # Determine device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Model is on device: {next(model.parameters()).device}, dtype: {dtype}")
    
    # Process each page with the VLM
    results = []
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing pages")):
        page_num = i + 1
        print(f"Processing page {page_num}/{len(image_paths)}")
        
        try:
            # Load and prepare the image
            pixel_values = load_image(image_path, max_num=max_num, dtype=dtype, device=device)
            print(f"Pixel values dtype: {pixel_values.dtype}, device: {pixel_values.device}")
            
            # Clear CUDA cache between pages to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
                
                # Process with smaller chunks if memory issues occur
                try:
                    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory when processing prompt. Falling back to CPU...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        model_cpu = model.cpu()
                        pixel_values_cpu = pixel_values.cpu().to(torch.float32)  # Use float32 on CPU
                        
                        # Process in smaller batches if needed
                        if pixel_values_cpu.size(0) > 1:
                            print(f"Processing in smaller batches (original batch size: {pixel_values_cpu.size(0)})")
                            responses = []
                            for k in range(0, pixel_values_cpu.size(0), batch_size):
                                batch = pixel_values_cpu[k:k+batch_size]
                                batch_response = model_cpu.chat(tokenizer, batch, prompt, generation_config)
                                responses.append(batch_response)
                            response = " ".join(responses)
                        else:
                            response = model_cpu.chat(tokenizer, pixel_values_cpu, prompt, generation_config)
                        
                        # Move model back to GPU if it was there before
                        if use_gpu and torch.cuda.is_available():
                            model.cuda()
                    else:
                        raise e
                
                if j == 0:
                    page_results["structure"] = response
                elif j == 1:
                    page_results["tables_lists"] = response
                elif j == 2:
                    page_results["figures"] = response
            
            page_results["page_number"] = page_num
            results.append(page_results)
            
            # Clear memory after each page
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing page {page_num}: {str(e)}")
            page_results = {
                "page_number": page_num,
                "error": str(e),
                "structure": "Error during processing",
                "tables_lists": "Error during processing",
                "figures": "Error during processing"
            }
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
            f.write(page.get('structure', 'Not available') + "\n\n")
            f.write("TABLES AND LISTS:\n")
            f.write(page.get('tables_lists', 'Not available') + "\n\n")
            f.write("FIGURES AND VISUALS:\n")
            f.write(page.get('figures', 'Not available') + "\n\n")
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
    pdf_path = "/content/SHIMI.pdf"  # Adjusted for Databricks DBFS
    model_path = "OpenGVLab/InternVL2-8B"
    output_dir = "/content/output"
    save_dir = "/content/results"
    model_save_dir = "/content/saved_model"
    max_num = 6  # Reduced from 12 to lower memory requirements
    
    # Memory optimization settings
    use_gpu = True  # Set to False to force CPU usage
    load_in_8bit = False  # Set to True to use 8-bit quantization
    load_in_4bit = False  # Set to True to use 4-bit quantization
    device_map = "auto"  # Helps with model distribution across multiple GPUs or CPU
    batch_size = 1  # For batch processing when falling back to smaller chunks
    
    # Determine available VRAM to guide loading strategy
    if use_gpu and torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Total GPU memory: {free_mem:.2f} GB")
        
        # Adjust settings based on available memory
        if free_mem < 10:  # Less than 10GB
            print("Low GPU memory detected, using 4-bit quantization")
            load_in_4bit = True
            max_num = 2  # Further reduce max tiles
        elif free_mem < 16:  # Less than 16GB
            print("Medium GPU memory detected, using 8-bit quantization")
            load_in_8bit = True
            max_num = 4  # Reduce max tiles
    
    # Load model with memory optimization
    try:
        model, tokenizer, dtype = load_model(
            model_path, 
            use_gpu=use_gpu, 
            device_map=device_map,
            load_in_8bit=load_in_8bit, 
            load_in_4bit=load_in_4bit
        )
        
        # Process PDF with memory management
        process_pdf_with_vlm(
            model=model,
            tokenizer=tokenizer,
            pdf_path=pdf_path,
            output_dir=output_dir,
            save_dir=save_dir,
            max_num=max_num,
            use_gpu=use_gpu,
            batch_size=batch_size,
            dtype=dtype
        )
        
        # Save model
        try:
            save_model_local(model, tokenizer, model_save_dir)
        except Exception as e:
            print(f"Warning: Could not save model locally due to: {e}")
            print("Skipping model saving step.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Try reducing max_num further or setting use_gpu=False to use CPU mode.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
