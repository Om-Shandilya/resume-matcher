import os
from pdfminer.high_level import extract_text
from docx import Document

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfminer.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    """
    Extracts text from a DOCX file using python-docx.

    Args:
        docx_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text from the DOCX.
    """
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_path):
    """
    Extracts text from a TXT file.
    
    Args:
        txt_path (str): Path to the TXT file.

    Returns:
        str: Extracted text from the TXT file.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_file(file_path):
    """
    Extracts text from a file based on its extension.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: Extracted text from the file.
    """
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        print(f"Extracting text from PDF {file_path}")
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        print(f"Extracting text from DOCX {file_path}")
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        print(f"Extracting text from TXT {file_path}")
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
