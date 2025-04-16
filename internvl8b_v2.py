import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import logging
from torchvision.transforms.functional import InterpolationMode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Image preprocessing functions from InternVL2 inference example
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

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class PDFParser:
    def __init__(self, model_name="OpenGVLab/InternVL2-8B", model_dir=None):
        """
        Initialize the PDF parser with the InternVL2-8B model.
        
        Args:
            model_name (str): HuggingFace model name
            model_dir (str, optional): Directory to save/load the model
        """
        logger.info(f"Initializing PDFParser with model: {model_name}")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
                logger.info(f"Loading model from local directory: {model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
                self.model = AutoModel.from_pretrained(
                    model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    use_flash_attn=True, trust_remote_code=True
                )
            else:
                logger.info(f"Downloading model from HuggingFace: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
                self.model = AutoModel.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    use_flash_attn=True, trust_remote_code=True
                )
                if model_dir:
                    logger.info(f"Saving model to directory: {model_dir}")
                    os.makedirs(model_dir, exist_ok=True)
                    self.tokenizer.save_pretrained(model_dir)
                    self.model.save_pretrained(model_dir)
                    logger.info(f"Model saved successfully to: {model_dir}")
            
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer not initialized properly")
            
            self.model = self.model.eval().to(self.device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def convert_pdf_to_images(self, pdf_path, output_dir=None, dpi=300):
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str, optional): Directory to save images
            dpi (int): DPI for image rendering
            
        Returns:
            list: List of image paths
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "pdf_images")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        images = []
        for page_num, page in enumerate(tqdm(doc, desc="Converting PDF pages")):
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img_path = os.path.join(output_dir, f"page_{page_num+1}.png")
            pix.save(img_path)
            images.append(img_path)
        logger.info(f"Converted {len(images)} PDF pages to images")
        return images

    def parse_image(self, image_path, prompt_template="Analyze this document page. Identify and extract all content, maintaining the hierarchical structure."):
        """
        Parse content from an image using InternVL2-8B model.
        
        Args:
            image_path (str): Path to the image
            prompt_template (str): Prompt for the model
            
        Returns:
            dict: Parsed content
        """
        logger.info(f"Parsing image: {image_path}")
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer not initialized")
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            pixel_values = load_image(image, input_size=448, max_num=12).to(torch.bfloat16).to(self.device)
            
            # Use model's chat method
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            response = self.model.chat(self.tokenizer, pixel_values, prompt_template, generation_config)
            parsed_text = response.strip()
            
            # Extract structured content
            structured_content = self._extract_structure(parsed_text)
            return structured_content
        except Exception as e:
            logger.error(f"Error parsing image {image_path}: {str(e)}")
            return {"error": str(e), "raw_text": "Failed to parse image"}

    def _extract_structure(self, text):
        """
        Extract structured information from the model's text output.
        
        Args:
            text (str): Text from the model
            
        Returns:
            dict: Structured content
        """
        try:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            result = {"content": []}
            current_section = None
            
            for para in paragraphs:
                if para.startswith('#'):
                    level = len(para) - len(para.lstrip('#'))
                    heading_text = para.lstrip('#').strip()
                    item = {"type": f"heading-{level}", "text": heading_text, "children": []}
                    result["content"].append(item)
                    current_section = item
                elif para.strip().startswith('- ') or para.strip().startswith('* '):
                    list_item = {"type": "list-item", "text": para.strip()[2:].strip()}
                    if current_section:
                        current_section["children"].append(list_item)
                    else:
                        result["content"].append(list_item)
                else:
                    para_item = {"type": "paragraph", "text": para.strip()}
                    if current_section:
                        current_section["children"].append(para_item)
                    else:
                        result["content"].append(para_item)
            return result
        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}")
            return {"raw_text": text}

    def parse_pdf(self, pdf_path, output_json_path=None, keep_images=False):
        """
        Parse a complete PDF document and maintain hierarchy.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_json_path (str, optional): Path to save JSON output
            keep_images (bool): Whether to keep the intermediate images
            
        Returns:
            dict: Complete parsed content
        """
        if output_json_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_json_path = f"{base_name}_parsed.json"
        
        temp_dir = os.path.join(os.path.dirname(pdf_path), f"{os.path.basename(pdf_path)}_temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        image_paths = []
        try:
            image_paths = self.convert_pdf_to_images(pdf_path, output_dir=temp_dir)
            result = {"document_name": os.path.basename(pdf_path), "pages": []}
            
            for i, img_path in enumerate(tqdm(image_paths, desc="Parsing pages")):
                prompt = f"This is page {i+1} of a document. Extract all content from this page, preserving the structure (headings, paragraphs, lists, tables)."
                page_content = self.parse_image(img_path, prompt)
                result["pages"].append({"page_number": i + 1, "content": page_content})
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"PDF parsing complete. Results saved to: {output_json_path}")
            return result
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            return {"error": str(e)}
        finally:
            if not keep_images and image_paths:
                for img_path in image_paths:
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary image: {img_path}. Error: {str(e)}")
                try:
                    os.rmdir(temp_dir)
                    logger.info(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory: {temp_dir}. Error: {str(e)}")

def main():
    config = {
        "pdf_files": ["/dbfs/path/to/your/document.pdf"],  # Update for Databricks DBFS
        "output_dir": "/dbfs/path/to/output/directory",
        "model_name": "OpenGVLab/InternVL2-8B",
        "model_dir": "/dbfs/path/to/model_dir",
        "keep_images": False,
        "dpi": 300
    }
    
    try:
        os.makedirs(config["model_dir"], exist_ok=True)
        logger.info(f"Initializing parser with model: {config['model_name']}")
        pdf_parser = PDFParser(model_name=config["model_name"], model_dir=config["model_dir"])
        
        os.makedirs(config["output_dir"], exist_ok=True)
        for pdf_path in config["pdf_files"]:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                continue
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_json_path = os.path.join(config["output_dir"], f"{base_name}_parsed.json")
            try:
                pdf_parser.parse_pdf(pdf_path=pdf_path, output_json_path=output_json_path, keep_images=config["keep_images"])
                logger.info(f"Successfully processed: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
