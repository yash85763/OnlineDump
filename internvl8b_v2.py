import os
import json
import torch
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, model_name="Internvl2/internvl2-8b"):
        """
        Initialize the PDF parser with the InternVL2-8B model.
        
        Args:
            model_name (str): HuggingFace model name for InternVL2
        """
        logger.info(f"Initializing PDFParser with model: {model_name}")
        
        # Load model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info("Loading model processor...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            logger.info("Loading model...")
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
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
            list: List of image paths or PIL Image objects
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
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and text
            inputs = self.processor(text=prompt_template, images=image, return_tensors="pt").to(self.device)
            
            # Generate content
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )
            
            # Decode the output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            parsed_text = generated_text.replace(prompt_template, "").strip()
            
            # Try to parse as structured content
            structured_content = self._extract_structure(parsed_text)
            return structured_content
        except Exception as e:
            logger.error(f"Error parsing image {image_path}: {str(e)}")
            return {"error": str(e), "raw_text": "Failed to parse image"}

    def _extract_structure(self, text):
        """
        Extract structured information from the model's text output.
        This is a placeholder - you may need to customize this based on 
        the model's output format and your specific needs.
        
        Args:
            text (str): Text from the model
            
        Returns:
            dict: Structured content
        """
        try:
            # Split by sections/paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Basic structure detection (headings, lists, paragraphs)
            result = {
                "content": []
            }
            
            current_section = None
            
            for para in paragraphs:
                if para.startswith('#'):  # Heading
                    # Count number of # to determine heading level
                    level = len(para) - len(para.lstrip('#'))
                    heading_text = para.lstrip('#').strip()
                    
                    item = {
                        "type": f"heading-{level}",
                        "text": heading_text,
                        "children": []
                    }
                    
                    result["content"].append(item)
                    current_section = item
                    
                elif para.strip().startswith('- ') or para.strip().startswith('* '):  # List item
                    list_item = {
                        "type": "list-item",
                        "text": para.strip()[2:].strip()
                    }
                    
                    if current_section:
                        current_section["children"].append(list_item)
                    else:
                        result["content"].append(list_item)
                        
                else:  # Regular paragraph
                    para_item = {
                        "type": "paragraph",
                        "text": para.strip()
                    }
                    
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
        # Generate output path if not provided
        if output_json_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_json_path = f"{base_name}_parsed.json"
        
        # Create temp directory for images
        temp_dir = os.path.join(os.path.dirname(pdf_path), f"{os.path.basename(pdf_path)}_temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        
        image_paths = []
        try:
            # Convert PDF to images
            image_paths = self.convert_pdf_to_images(pdf_path, output_dir=temp_dir)
            
            # Parse each page
            result = {
                "document_name": os.path.basename(pdf_path),
                "pages": []
            }
            
            for i, img_path in enumerate(tqdm(image_paths, desc="Parsing pages")):
                # Create a page-specific prompt
                prompt = f"This is page {i+1} of a document. Extract all content from this page, preserving the structure (headings, paragraphs, lists, tables). Identify section headings, paragraphs, bullet points, and tables."
                
                # Parse the page
                page_content = self.parse_image(img_path, prompt)
                
                result["pages"].append({
                    "page_number": i + 1,
                    "content": page_content
                })
            
            # Save result to JSON
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"PDF parsing complete. Results saved to: {output_json_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            return {"error": str(e)}
        finally:
            # Clean up temporary image files if not keeping them
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


# Main execution code - specify files directly here
def main():
    # Configuration
    config = {
        "pdf_files": [
            # List your PDF files here
            "/path/to/your/document1.pdf",
            "/path/to/your/document2.pdf"
        ],
        "output_dir": "/path/to/output/directory",  # Output directory for JSON files
        "model_name": "Internvl2/internvl2-8b",     # Model to use
        "keep_images": False,                       # Whether to keep intermediate images
        "dpi": 300                                  # DPI for PDF rendering
    }
    
    try:
        # Initialize parser
        logger.info(f"Initializing parser with model: {config['model_name']}")
        pdf_parser = PDFParser(model_name=config["model_name"])
        
        # Ensure output directory exists
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Process each PDF file
        for pdf_path in config["pdf_files"]:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                continue
                
            # Generate output JSON path
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_json_path = os.path.join(config["output_dir"], f"{base_name}_parsed.json")
            
            # Parse PDF
            try:
                pdf_parser.parse_pdf(
                    pdf_path=pdf_path,
                    output_json_path=output_json_path,
                    keep_images=config["keep_images"]
                )
                logger.info(f"Successfully processed: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
