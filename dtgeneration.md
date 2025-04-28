Contract Analysis Viewer
Overview
The Contract Analysis Viewer is a Streamlit-based web application designed to analyze and visualize PDF contracts. It allows users to upload PDFs or select pre-loaded contracts, extract text, analyze content, and display results in a user-friendly interface. The application supports OCR’d PDFs, handles large files with warnings, and provides error handling for robust operation. Key features include:

Three-Pane Interface: 
Left Pane: PDF upload, pre-loaded PDF selection, analysis controls, and status display (scrollable).
Middle Pane: PDF viewer with iframe/object tag rendering and search/highlight functionality.
Right Pane: Displays analysis results (form number, summary, contract status, and clauses).


Pre-loaded PDFs and JSONs: Loads PDFs from ./preloaded_contracts/pdfs and JSONs from ./preloaded_contracts/jsons, with automatic folder creation and fallback to the current directory if files are missing.
Processing Messages: Displays "Text extracted from PDF" and "Analyzing the document" under a spinner during analysis.
Error Handling: Retries analysis up to two times, logs errors via ECFRLogger, and provides user feedback.
Large PDF Support: Warns for PDFs >1.5MB due to base64 encoding limitations (noted April 24, 2025).
OCR Support: Compatible with image-based PDFs (noted April 14, 2025).

Prerequisites
Before running the application, ensure the following are installed and configured:
Software Requirements

Python: Version 3.8 or higher.
pip: Python package manager.
Virtual Environment (recommended): To isolate dependencies (e.g., venv or virtualenv).
Operating System: Windows, macOS, or Linux.

Hardware Requirements

RAM: Minimum 4GB (8GB recommended for processing large PDFs).
Disk Space: At least 500MB for dependencies and temporary files.

Required Python Libraries

streamlit: For the web interface.
PyPDF2: For PDF processing.
Custom classes (provided by the user):
PDFTextProcessor: Extracts text from PDFs.
ECFRLogger: Handles logging (info, warning, error methods).
ContractAnalyzer: Analyzes contract text and generates JSON.



Installation
Follow these steps to set up the environment and install dependencies.
Step 1: Clone or Download the Project

If the project is in a repository, clone it:git clone <repository-url>
cd <project-directory>


Alternatively, download and extract the project files to a local directory.

Step 2: Set Up a Virtual Environment
Create and activate a virtual environment to isolate dependencies.

On Windows:python -m venv venv
.\venv\Scripts\activate


On macOS/Linux:python3 -m venv venv
source venv/bin/activate



You should see (venv) in your terminal prompt, indicating the virtual environment is active.
Step 3: Install Dependencies
Install the required Python libraries using pip:
pip install streamlit PyPDF2

Step 4: Provide Custom Classes
Ensure the following custom classes are available in the project directory or Python path:

pdf_text_processor.py:
Contains PDFTextProcessor with a process_pdf(pdf_path, preprocessed_path) method that extracts text, saves it to preprocessed_path, and returns the text.


ecfr_logger.py:
Contains ECFRLogger with info, warning, and error methods for logging.


contract_analyzer.py:
Contains ContractAnalyzer with an analyze_contract(text, output_path) method that analyzes text, saves JSON to output_path, and returns results.



Example Directory Structure:
project-directory/
├── streamlit.py
├── pdf_text_processor.py
├── ecfr_logger.py
├── contract_analyzer.py

If these classes are not provided, implement stubs or contact the project maintainer for the actual implementations. Example stub for PDFTextProcessor:
class PDFTextProcessor:
    def process_pdf(self, pdf_path, preprocessed_path):
        # Stub: Replace with actual text extraction logic
        with open(preprocessed_path, 'w') as f:
            f.write("Sample extracted text")
        return "Sample extracted text"

Step 5: Set Up Folder Structure (Optional)
The application automatically creates the following folders if they don’t exist:

./preloaded_contracts/pdfs: For pre-loaded PDFs.
./preloaded_contracts/jsons: For pre-loaded JSON analysis files.

If you want to pre-populate these folders:

Place PDF files (e.g., contract1.pdf, contract2.pdf) in ./preloaded_contracts/pdfs.
Place corresponding JSON files (e.g., contract1.json, contract2.json) in ./preloaded_contracts/jsons, ensuring filenames match by stem (e.g., contract1.pdf pairs with contract1.json).

Example:
project-directory/
├── preloaded_contracts/
│   ├── pdfs/
│   │   ├── contract1.pdf
│   │   ├── contract2.pdf
│   ├── jsons/
│   │   ├── contract1.json
│   │   ├── contract2.json

If no files are found in these folders, the application searches the current working directory for preloaded_contracts/pdfs and preloaded_contracts/jsons.
Running the Application
Follow these steps to start the Streamlit application.
Step 1: Verify the Main Script
Ensure streamlit.py is in the project directory and contains the application code.
Step 2: Activate the Virtual Environment
If not already activated:

Windows:.\venv\Scripts\activate


macOS/Linux:source venv/bin/activate



Step 3: Run the Streamlit Application
Execute the following command from the project directory:
streamlit run streamlit.py

Step 4: Access the Application

Streamlit will start a local server and open the application in your default web browser.
The URL is typically http://localhost:8501.
If the browser doesn’t open automatically, navigate to http://localhost:8501 manually.

Step 5: Interact with the Application

Left Pane:
Use the "Pre-loaded PDFs" dropdown to select a single PDF or "Load all pre-loaded PDFs".
Upload new PDFs via the "Upload PDFs" section.
Click "Analyze PDFs" to process unanalyzed PDFs.
View analysis status and navigate pages.


Middle Pane:
View the selected PDF with search/highlight functionality.


Right Pane:
Review analysis results (form number, summary, contract status, clauses).



Troubleshooting
Common issues and solutions:
Installation Issues

pip Install Fails:
Ensure pip is up-to-date:pip install --upgrade pip


Verify internet connectivity and try again.


Module Not Found:
Confirm streamlit and PyPDF2 are installed in the active virtual environment:pip show streamlit
pip show PyPDF2


Reinstall if missing:pip install streamlit PyPDF2





Folder and File Issues

Folders Not Created:
Verify write permissions in the project directory:ls -ld .


Check logs for errors (logged via ECFRLogger).


No PDFs/JSONs Found:
Ensure PDFs are in ./preloaded_contracts/pdfs or {current_dir}/preloaded_contracts/pdfs.
Ensure JSONs are in ./preloaded_contracts/jsons or {current_dir}/preloaded_contracts/jsons.
Verify filenames match by stem (e.g., contract1.pdf and contract1.json).
Check logs:logger.error(f"No PDF files found in {pdf_folder}")
logger.warning(f"No JSON files found in {json_folder}. No preprocessed data available.")


Inspect folders:ls -l ./preloaded_contracts/pdfs
ls -l ./preloaded_contracts/jsons





Application Issues

Streamlit Fails to Start:
Ensure port 8501 is free:netstat -an | findstr 8501  # Windows
lsof -i :8501              # macOS/Linux


Run with a different port if needed:streamlit run streamlit.py --server.port 8502




Custom Classes Missing:
Verify pdf_text_processor.py, ecfr_logger.py, and contract_analyzer.py are in the project directory.
Check Python path:python -c "import sys; print(sys.path)"




Processing Messages Not Showing:
Confirm ECFRLogger logs info messages:logger.info(f"Text extracted from {pdf_name}")


Verify Streamlit rendering:message_placeholder.markdown(...)





PDF and Analysis Issues

Large PDFs Fail to Display:
PDFs >1.5MB may fail due to base64 encoding (noted April 24, 2025).
Use the download button to access the PDF.
Compress PDFs with qpdf:qpdf --stream-data=compress input.pdf output.pdf




Image-Based PDFs Fail:
Ensure PDFTextProcessor supports OCR (noted April 14, 2025).
Preprocess with Tesseract if needed:tesseract input.pdf output -l eng pdf




Analysis Errors:
Check logs for ContractAnalyzer retries:logger.error(f"Failed after {max_retries} retries for {pdf_name}: {str(e)}")


Verify ContractAnalyzer implementation.



Testing Instructions
To ensure the application works as expected, perform the following tests:

Folder Creation:

Delete ./preloaded_contracts/pdfs and ./preloaded_contracts/jsons.
Run the app and verify folders are created.
Test with {current_dir}/preloaded_contracts/pdfs and jsons.


No PDFs/JSONs Handling:

Ensure ./preloaded_contracts/pdfs is empty.
Verify:
Dropdown shows "No pre-loaded PDFs available".
Warning: "No pre-loaded PDFs found in ./preloaded_contracts/pdfs or {current_dir}/preloaded_contracts/pdfs."
Log: ERROR: No PDF files found in {path}.


Place PDFs in {current_dir}/preloaded_contracts/pdfs and confirm they load.
Repeat for JSONs.


Pre-loaded PDFs Dropdown:

Add PDFs (contract1.pdf, contract2.pdf) to ./preloaded_contracts/pdfs and JSONs (contract1.json) to ./preloaded_contracts/jsons.
Verify dropdown lists PDFs, "Select a pre-loaded PDF", and "Load all pre-loaded PDFs".
Select "Load all pre-loaded PDFs" and confirm:
PDFs appear in "Available PDFs".
PDFs with JSONs are "Processed"; others are "Not processed".
Success message: "Loaded pre-loaded PDFs: contract1.pdf, contract2.pdf".




Processing Messages:

Analyze a PDF without a JSON.
Verify:
Spinner shows Processing {pdf_name}....
Messages: "Text extracted from PDF", "Analyzing the document".
Messages are blue, 14px, and scroll in the left pane.




Scrollable Panes:

Load multiple PDFs and confirm left pane scrolling.


Error Handling:

Test with a PDF causing ContractAnalyzer errors.
Verify retries and error: "An error occurred. Please submit another PDF or reach out to Support."


PDF Display and JSON Results:

Verify PDF rendering and JSON data (form number, summary, clauses).


Search Functionality:

Test clause search/highlight.



Contributing
To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License (or specify your license).
Contact
For issues or questions, contact the project maintainer or open an issue in the repository.
