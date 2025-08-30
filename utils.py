from pathlib import Path
import json


def write_json(data: dict, outfile: str | Path, *, indent: int = 4) -> Path:
    """
    Serialize `data` to a JSON file at `outfile`.

    Parameters
    ----------
    data     : dict
        The dictionary (or any JSON-serialisable object) to write.
    outfile  : str | pathlib.Path
        Destination file path. Parent directories are created if missing.
    indent   : int, default=4
        Indentation level for pretty printing. Pass None for compact format.

    Returns
    -------
    pathlib.Path
        The absolute path of the file that was written.
    """
    out_path = Path(outfile).expanduser().resolve()

    # Ensure the parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON with UTF-8 encoding
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return out_path

def load_json_to_dict(file_path):
    """
    Load JSON file to dictionary.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON as dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def write_markdown(content: str, filepath: str, encoding: str = 'utf-8') -> None:
    """
    Write markdown content string to a .md file.
    
    Args:
        content (str): Markdown content as string
        filepath (str): Path where to save the .md file
        encoding (str): File encoding (default: utf-8)
    """
    try:
        with open(filepath, 'w', encoding=encoding) as file:
            file.write(content)
        print(f"Markdown file saved successfully: {filepath}")
    except Exception as e:
        print(f"Error writing markdown file: {e}")

import os

def find_pdf_docx_files(directory):
    """Find all PDF and DOCX files in directory and subdirectories"""
    pdf_docx_files = []
    
    # Walk through directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx')):
                full_path = os.path.join(root, file)
                pdf_docx_files.append(full_path)
    
    return pdf_docx_files