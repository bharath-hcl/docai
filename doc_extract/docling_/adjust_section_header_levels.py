"""
Docling Section Header Level Adjustment Module

This module processes enriched Docling JSON output to:
1. Identify section headers based on font size hierarchy
2. Optionally filter out non-bold section headers
3. Assign appropriate level values (0 = largest font, 1, 2, 3... = smaller fonts)
"""

from collections import Counter
from typing import Dict, Any, List, Optional, Tuple
import copy

def get_mode(values: List[Any]) -> Optional[Any]:
    """
    Get the most frequent value (mode) from a list.
    
    Args:
        values: List of values to find mode for
    
    Returns:
        Most frequent value or None if list is empty
    """
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]

def is_font_bold(span: Dict[str, Any]) -> bool:
    """
    Determine if a text span represents bold text.
    
    Args:
        span: Text span dictionary with font metadata
    
    Returns:
        True if span represents bold text, False otherwise
    """
    # Check is_bold flag first (most reliable)
    if span.get('is_bold', False):
        return True
    
    # Check font name for bold indicators as backup
    font_name = span.get('font', '').lower()
    bold_indicators = ['bold', 'black', 'heavy', 'thick', 'medium', 'semibold', 'extrabold']
    return any(indicator in font_name for indicator in bold_indicators)

def extract_font_info(text_meta: List[Dict[str, Any]]) -> Tuple[Optional[float], bool]:
    """
    Extract dominant font size and bold status from text_meta spans.
    
    Args:
        text_meta: List of text span dictionaries
    
    Returns:
        Tuple of (mode_font_size, is_predominantly_bold)
    """
    if not text_meta:
        return None, False
    
    font_sizes = []
    bold_statuses = []
    
    for span in text_meta:
        # Extract font size if available
        if 'size' in span and span['size'] is not None:
            font_sizes.append(float(span['size']))
        
        # Extract bold status
        bold_statuses.append(is_font_bold(span))
    
    # Calculate modes
    mode_size = get_mode(font_sizes) if font_sizes else None
    mode_bold = get_mode(bold_statuses) if bold_statuses else False
    
    return mode_size, mode_bold

def collect_section_headers(texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collect all section headers with their font information.
    
    Args:
        texts: List of text elements from Docling JSON
    
    Returns:
        List of header info dictionaries
    """
    section_headers = []
    
    for text_elem in texts:
        if text_elem.get('label') == 'section_header':
            text_meta = text_elem.get('text_meta', [])
            font_size, is_bold = extract_font_info(text_meta)
            
            section_headers.append({
                'element': text_elem,
                'font_size': font_size,
                'is_bold': is_bold,
                'text': text_elem.get('text', '')[:50] + '...' if len(text_elem.get('text', '')) > 50 else text_elem.get('text', '')
            })
    
    return section_headers

def filter_non_bold_headers(section_headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out non-bold headers and remove their section_header labels.
    
    Args:
        section_headers: List of header info dictionaries
    
    Returns:
        List of bold headers only
    """
    bold_headers = []
    removed_headers = []
    
    for header_info in section_headers:
        if header_info['is_bold']:
            bold_headers.append(header_info)
        else:
            # Remove section_header label from non-bold elements
            element = header_info['element']
            if 'label' in element:
                del element['label']
            if 'level' in element:
                del element['level']
            removed_headers.append(header_info['text'])
    
    if removed_headers:
        print(f"Removed {len(removed_headers)} non-bold section headers:")
        for text in removed_headers:
            print(f"  - {text}")
    
    return bold_headers

def assign_header_levels(headers: List[Dict[str, Any]]) -> None:
    """
    Assign hierarchical levels based on font sizes.
    
    Args:
        headers: List of header info dictionaries
    """
    if not headers:
        return
    
    # Get unique font sizes and sort in descending order (largest first)
    valid_headers = [h for h in headers if h['font_size'] is not None]
    
    if not valid_headers:
        print("Warning: No headers with valid font sizes found")
        return
    
    unique_sizes = sorted(set(h['font_size'] for h in valid_headers), reverse=True)
    
    # Create size to level mapping (largest size = level 0)
    size_to_level = {size: level for level, size in enumerate(unique_sizes)}
    
    print(f"Font size hierarchy (size -> level):")
    for size, level in size_to_level.items():
        print(f"  {size}pt -> Level {level}")
    
    # Assign levels to headers
    level_counts = Counter()
    for header_info in valid_headers:
        level = size_to_level[header_info['font_size']]
        header_info['element']['level'] = level
        level_counts[level] += 1
    
    print(f"Assigned levels to {len(valid_headers)} headers:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} headers")

def adjust_section_header_levels(docling_dict: Dict[str, Any], remove_non_bold: bool = False) -> Dict[str, Any]:
    """
    Main function to adjust section header levels based on font size hierarchy
    and optionally filter non-bold headers.
    
    Args:
        docling_dict: Dictionary from enrich_docling_with_pymupdf_fixed output
        remove_non_bold: If True, remove section_header labels for non-bold headers
    
    Returns:
        Updated dictionary with adjusted header levels
    """
    # Create a deep copy to avoid modifying the original
    result_dict = copy.deepcopy(docling_dict)
    
    # Get text elements
    texts = result_dict.get('texts', [])
    if not texts:
        print("Warning: No text elements found in input dictionary")
        return result_dict
    
    print("=== ADJUSTING SECTION HEADER LEVELS ===")
    
    # Step 1: Collect all section headers
    section_headers = collect_section_headers(texts)
    print(f"Found {len(section_headers)} section headers")
    
    if not section_headers:
        print("No section headers found to process")
        return result_dict
    
    # Step 2: Optionally filter out non-bold headers
    if remove_non_bold:
        processed_headers = filter_non_bold_headers(section_headers)
        print(f"Kept {len(processed_headers)} bold section headers")
    else:
        processed_headers = section_headers
        print("Keeping all section headers regardless of boldness")
    
    # Step 3: Assign hierarchical levels based on font sizes
    assign_header_levels(processed_headers)
    
    print("=== HEADER LEVEL ADJUSTMENT COMPLETE ===")
    return result_dict

def process_docling_headers(docling_dict: Dict[str, Any], verbose: bool = True, remove_non_bold: bool = False) -> Dict[str, Any]:
    """
    Main interface function to process Docling dictionary and adjust section header levels.
    
    Args:
        docling_dict: Dictionary from enrich_docling_with_pymupdf_fixed output
        verbose: Whether to print processing information
        remove_non_bold: If True, remove section_header labels for non-bold headers (default: False)
    
    Returns:
        Updated dictionary with adjusted header levels and optionally filtered headers
    """
    if not verbose:
        # Temporarily redirect print statements
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            result = adjust_section_header_levels(docling_dict, remove_non_bold=remove_non_bold)
        finally:
            sys.stdout = old_stdout
        return result
    else:
        return adjust_section_header_levels(docling_dict, remove_non_bold=remove_non_bold)

# Utility function for debugging
def analyze_headers(docling_dict: Dict[str, Any]) -> None:
    """
    Analyze section headers in the document for debugging purposes.
    
    Args:
        docling_dict: Dictionary from enrich_docling_with_pymupdf_fixed output
    """
    texts = docling_dict.get('texts', [])
    headers = collect_section_headers(texts)
    
    print("=== HEADER ANALYSIS ===")
    print(f"Total section headers found: {len(headers)}")
    
    if headers:
        print("\nHeader details:")
        for i, header in enumerate(headers):
            print(f"{i+1}. {header['text']}")
            print(f"   Font size: {header['font_size']}")
            print(f"   Is bold: {header['is_bold']}")
            print(f"   Current level: {header['element'].get('level', 'Not set')}")
            
            # Show some font details from text_meta
            text_meta = header['element'].get('text_meta', [])
            if text_meta:
                fonts = set(span.get('font', 'Unknown') for span in text_meta)
                sizes = set(span.get('size', 'Unknown') for span in text_meta)
                print(f"   Fonts: {', '.join(fonts)}")
                print(f"   Sizes: {', '.join(map(str, sizes))}")
            print()

if __name__ == "__main__":
    # Example usage
    print("Docling Section Header Level Adjustment Module")
    print("Import this module and use process_docling_headers(docling_dict, remove_non_bold=False)")
    print("For debugging, use analyze_headers(docling_dict)")
