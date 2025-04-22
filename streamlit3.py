import os
import json
import base64
import streamlit as st
from pathlib import Path
import pandas as pd
import re
import urllib.parse
import fitz  # PyMuPDF for PDF validation

# Set page configuration
st.set_page_config(
    page_title="Research Paper Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-container {
        display: flex;
        flex-direction: row;
    }
    .pdf-viewer {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
    }
    .json-details {
        border: 1px solid #ddd;
        border-radius: 5
