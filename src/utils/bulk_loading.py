import os
from typing import List, Dict, Union

from src.processing.text_cleaning import clean_text  # assuming you already have this
from src.utils.file_reader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt

def bulk_load_cleaned_resume_files(input_path: Union[str, List[str]]) -> Dict[str, str]:
    """
    Load multiple resumes from a directory or list of files.

    Args:
        input_path : str or List[str]
            Either:
            - A path to a directory containing resume files, OR
            - A list of individual file paths.

    Returns:
        Dict[str, str]: Dictionary mapping file's basenames -> cleaned text.
    """
    resumes = {}

    # Case 1: directory
    if isinstance(input_path, str) and os.path.isdir(input_path):
        file_paths = [
            os.path.join(input_path, f) 
            for f in os.listdir(input_path)
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        ]
    # Case 2: list of files
    elif isinstance(input_path, list):
        file_paths = input_path
    # Case 3: single file
    elif isinstance(input_path, str) and os.path.isfile(input_path):
        file_paths = [input_path]
    else:
        raise ValueError(f"Invalid input_path: {input_path}")

    # Extract text from each file
    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        text = ""
        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext == ".docx":
                text = extract_text_from_docx(path)
            elif ext == ".txt":
                text = extract_text_from_txt(path)
            else:
                print(f"⚠️ Skipping unsupported file type: {path}")
                continue

            cleaned = clean_text(text)
            resumes[os.path.basename(path)] = cleaned

        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    print(f"✅ Loaded {len(resumes)} resumes.")

    return resumes
