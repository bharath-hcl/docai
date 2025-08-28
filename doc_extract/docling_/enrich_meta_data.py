#!/usr/bin/env python3
"""
Corrected script to enrich Docling JSON output with PyMuPDF text metadata.
Fixed coordinate conversion and bounding box ordering.
"""

import json
import fitz  # PyMuPDF
import sys
import os
from typing import List, Dict, Any, Optional

def pymupdf_bbox_to_docling_bbox_fixed(pymupdf_bbox: tuple, page_height: float) -> Dict[str, float]:
    """
    Convert PyMuPDF bounding box format to Docling format with proper coordinate handling.
    
    PyMuPDF: (x0, y0, x1, y1) where y0 < y1 in PDF coordinate system
    Docling: {l, t, r, b} where t < b in their coordinate system
    """
    x0, y0, x1, y1 = pymupdf_bbox
    
    # Convert y-coordinates: PDF origin is bottom-left, we need top-left reference
    # The smaller y value in PDF becomes the larger t value in Docling
    # The larger y value in PDF becomes the smaller b value in Docling
    l = x0
    t = page_height - y1  # y1 is the top in PDF coords
    r = x1
    b = page_height - y0  # y0 is the bottom in PDF coords
    
    # Ensure proper ordering: t should be < b in Docling coordinate system
    if t > b:
        t, b = b, t
        
    return {"l": l, "t": t, "r": r, "b": b}


def calculate_bbox_overlap_percentage(span_bbox: Dict[str, float], docling_bbox: Dict[str, float]) -> float:
    """Calculate the percentage of span_bbox that overlaps with docling_bbox."""
    # Ensure both bboxes have proper ordering
    span_bbox = ensure_bbox_ordering(span_bbox)
    docling_bbox = ensure_bbox_ordering(docling_bbox)
    
    # Calculate intersection boundaries
    x_left = max(span_bbox["l"], docling_bbox["l"])
    y_top = max(span_bbox["t"], docling_bbox["t"])
    x_right = min(span_bbox["r"], docling_bbox["r"])
    y_bottom = min(span_bbox["b"], docling_bbox["b"])
    
    # Check if there's no overlap
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    span_area = (span_bbox["r"] - span_bbox["l"]) * (span_bbox["b"] - span_bbox["t"])
    
    if span_area <= 0:
        return 0.0
    
    return intersection_area / span_area


def ensure_bbox_ordering(bbox: Dict[str, float]) -> Dict[str, float]:
    """Ensure bbox coordinates are in proper order (t < b, l < r)."""
    l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
    
    if l > r:
        l, r = r, l
    if t > b:
        t, b = b, t
        
    return {"l": l, "t": t, "r": r, "b": b}


def extract_pdf_spans_fixed(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract all text spans from PDF using PyMuPDF with fixed coordinate conversion."""
    pdf_document = fitz.open(pdf_path)
    spans = []
    
    print(f"PDF has {len(pdf_document)} pages")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        page_height = page.rect.height
        print(f"Page {page_num + 1}: height = {page_height}")
        
        # Get text dictionary with detailed information
        text_dict = page.get_text("dict")
        page_span_count = 0
        
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block (not image)
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Convert bounding box to Docling coordinate system with fixed conversion
                        docling_bbox = pymupdf_bbox_to_docling_bbox_fixed(span["bbox"], page_height)
                        
                        span_info = {
                            "page_no": page_num + 1,
                            "text": span["text"],
                            "font": span["font"],
                            "size": round(span["size"], 2),
                            "flags": span["flags"],
                            "bbox": docling_bbox,
                            "color": span.get("color", 0),
                            "original_pymupdf_bbox": span["bbox"]  # Keep for debugging
                        }
                        
                        # Add font style information based on flags
                        span_info["is_bold"] = bool(span["flags"] & 2**4)
                        span_info["is_italic"] = bool(span["flags"] & 2**1)
                        span_info["is_superscript"] = bool(span["flags"] & 2**0)
                        span_info["is_subscript"] = bool(span["flags"] & 2**2)
                        
                        spans.append(span_info)
                        page_span_count += 1
        
        print(f"Page {page_num + 1}: extracted {page_span_count} spans")
    
    pdf_document.close()
    return spans


def match_spans_to_docling_elements_fixed(docling_json: Dict[str, Any], pdf_spans: List[Dict[str, Any]], 
                                        overlap_threshold: float = 0.5) -> Dict[str, Any]:
    """Match PDF spans to Docling text elements with fixed coordinate handling."""
    print(f"\n=== MATCHING SPANS WITH THRESHOLD {overlap_threshold} ===")
    
    # Group spans by page for easier debugging
    spans_by_page = {}
    for span in pdf_spans:
        page_no = span["page_no"]
        if page_no not in spans_by_page:
            spans_by_page[page_no] = []
        spans_by_page[page_no].append(span)
    
    print(f"PDF spans by page: {[(page, len(spans)) for page, spans in spans_by_page.items()]}")
    
    # Process all text elements
    total_matches = 0
    for idx, text_elem in enumerate(docling_json.get("texts", [])):
        print(f"\n--- Processing Docling text element {idx} ---")
        
        # Get provenance information
        prov = text_elem.get("prov", [])
        if not prov:
            print("  No provenance info, skipping")
            text_elem["text_meta"] = []
            continue
            
        # Get page number and bounding box
        page_no = prov[0].get("page_no")
        docling_bbox = prov[0].get("bbox")
        
        print(f"  Docling text: '{text_elem.get('text', '')[:50]}...'")
        print(f"  Page: {page_no}")
        print(f"  Bbox: {docling_bbox}")
        
        if page_no is None or docling_bbox is None:
            print("  Missing page_no or bbox, skipping")
            text_elem["text_meta"] = []
            continue
        
        # Find matching spans on the same page
        matched_spans = []
        page_spans = spans_by_page.get(page_no, [])
        print(f"  Checking against {len(page_spans)} spans on page {page_no}")
        
        best_overlaps = []
        for span in page_spans:
            overlap = calculate_bbox_overlap_percentage(span["bbox"], docling_bbox)
            if overlap > 0:
                best_overlaps.append((overlap, span["text"][:30]))
            
            if overlap >= overlap_threshold:
                # Create a copy of span for the match (remove original_pymupdf_bbox from output)
                matched_span = {
                    "text": span["text"],
                    "font": span["font"],
                    "size": span["size"],
                    "flags": span["flags"],
                    "color": span["color"],
                    "is_bold": span["is_bold"],
                    "is_italic": span["is_italic"],
                    "is_superscript": span["is_superscript"],
                    "is_subscript": span["is_subscript"],
                    "bbox": span["bbox"],
                    "overlap_percentage": round(overlap, 3)
                }
                matched_spans.append(matched_span)
        
        # Show top overlaps for debugging
        best_overlaps.sort(reverse=True, key=lambda x: x[0])
        if best_overlaps:
            print(f"  Top overlaps: {best_overlaps[:5]}")  # Show top 5
        else:
            print("  No overlaps found!")
        
        # Sort matches by overlap percentage
        matched_spans.sort(key=lambda s: (-s["overlap_percentage"], s["bbox"]["t"], s["bbox"]["l"]))
        
        print(f"  Matched {len(matched_spans)} spans")
        if matched_spans:
            total_matches += 1
        
        # Add metadata to the text element
        text_elem["text_meta"] = matched_spans
    
    print(f"\n=== SUMMARY: {total_matches}/{len(docling_json.get('texts', []))} elements matched ===")
    return docling_json


def enrich_docling_with_pymupdf(docling_json: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """Main function with fixed coordinate conversion."""
    # Validate input files
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Docling JSON has {len(docling_json.get('texts', []))} text elements")
    
    print(f"\nExtracting text spans from PDF: {pdf_path}")
    
    # Extract spans from PDF with fixed coordinate conversion
    pdf_spans = extract_pdf_spans_fixed(pdf_path)
    print(f"Total extracted spans: {len(pdf_spans)}")
    
    # Match spans to Docling elements with 50% threshold (should work well now)
    print("\n=== MATCHING PHASE ===")
    enriched_json = match_spans_to_docling_elements_fixed(docling_json, pdf_spans, overlap_threshold=0.5)
    
    print("Enrichment completed!")
    return enriched_json

def match_spans_to_docling_elements_with_reporting(docling_json: Dict[str, Any], pdf_spans: List[Dict[str, Any]], 
                                                 overlap_threshold: float = 0.5) -> Dict[str, Any]:
    """Match PDF spans to Docling text elements and report unmatched elements."""
    print(f"\n=== MATCHING SPANS WITH THRESHOLD {overlap_threshold} ===")
    
    # Group spans by page for easier debugging
    spans_by_page = {}
    for span in pdf_spans:
        page_no = span["page_no"]
        if page_no not in spans_by_page:
            spans_by_page[page_no] = []
        spans_by_page[page_no].append(span)
    
    print(f"PDF spans by page: {[(page, len(spans)) for page, spans in spans_by_page.items()]}")
    
    # Process all text elements and track unmatched ones
    total_matches = 0
    unmatched_elements = []
    
    for idx, text_elem in enumerate(docling_json.get("texts", [])):
        # Get provenance information
        prov = text_elem.get("prov", [])
        if not prov:
            text_elem["text_meta"] = []
            unmatched_elements.append((None, text_elem.get('text', '<no text>'), 'No provenance info'))
            continue
            
        # Get page number and bounding box
        page_no = prov[0].get("page_no")
        docling_bbox = prov[0].get("bbox")
        
        if page_no is None or docling_bbox is None:
            text_elem["text_meta"] = []
            unmatched_elements.append((page_no, text_elem.get('text', '<no text>'), 'Missing page_no or bbox'))
            continue
        
        # Find matching spans on the same page
        matched_spans = []
        page_spans = spans_by_page.get(page_no, [])
        
        best_overlap = 0.0
        for span in page_spans:
            overlap = calculate_bbox_overlap_percentage(span["bbox"], docling_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
            
            if overlap >= overlap_threshold:
                # Create a copy of span for the match
                matched_span = {
                    "text": span["text"],
                    "font": span["font"],
                    "size": span["size"],
                    "flags": span["flags"],
                    "color": span["color"],
                    "is_bold": span["is_bold"],
                    "is_italic": span["is_italic"],
                    "is_superscript": span["is_superscript"],
                    "is_subscript": span["is_subscript"],
                    "bbox": span["bbox"],
                    "overlap_percentage": round(overlap, 3)
                }
                matched_spans.append(matched_span)
        
        # Sort matches by overlap percentage
        matched_spans.sort(key=lambda s: (-s["overlap_percentage"], s["bbox"]["t"], s["bbox"]["l"]))
        
        if matched_spans:
            total_matches += 1
        else:
            # Track unmatched element with best overlap found
            reason = f'Best overlap: {best_overlap:.3f} (below threshold {overlap_threshold})'
            if len(page_spans) == 0:
                reason = 'No spans found on this page'
            unmatched_elements.append((page_no, text_elem.get('text', '<no text>'), reason))
        
        # Add metadata to the text element
        text_elem["text_meta"] = matched_spans
    
    # Report results
    total_elements = len(docling_json.get("texts", []))
    print(f"\n=== MATCHING RESULTS ===")
    print(f"Matched: {total_matches}/{total_elements} elements")
    print(f"Unmatched: {len(unmatched_elements)} elements")
    
    if unmatched_elements:
        print(f"\n=== UNMATCHED ELEMENTS ({len(unmatched_elements)}) ===")
        for page_no, text, reason in unmatched_elements:
            # Truncate long text for readability
            truncated_text = text[:100] + "..." if len(text) > 100 else text
            print(f"Page {page_no}: {truncated_text}")
            print(f"  Reason: {reason}\n")
        
        # Summary by page
        page_counts = {}
        for page_no, _, _ in unmatched_elements:
            if page_no is not None:
                page_counts[page_no] = page_counts.get(page_no, 0) + 1
        
        if page_counts:
            print("=== UNMATCHED BY PAGE ===")
            for page, count in sorted(page_counts.items()):
                print(f"Page {page}: {count} unmatched elements")
    
    return docling_json


def enrich_docling_with_pymupdf_with_reporting(docling_json_path: str, pdf_path: str, output_path: Optional[str] = None) -> None:
    """Main function with detailed reporting of unmatched elements."""
    # ... [previous validation code remains the same] ...
    
    if output_path is None:
        base_name = os.path.splitext(docling_json_path)[0]
        output_path = f"{base_name}_enriched.json"
    
    print(f"Loading Docling JSON from: {docling_json_path}")
    
    # Load Docling JSON
    try:
        with open(docling_json_path, "r", encoding="utf-8") as f:
            docling_json = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in Docling file: {e}")
    
    print(f"Docling JSON has {len(docling_json.get('texts', []))} text elements")
    
    print(f"\nExtracting text spans from PDF: {pdf_path}")
    
    # Extract spans from PDF with fixed coordinate conversion
    pdf_spans = extract_pdf_spans_fixed(pdf_path)
    print(f"Total extracted spans: {len(pdf_spans)}")
    
    # Match spans to Docling elements with detailed reporting
    print("\n=== MATCHING PHASE ===")
    enriched_json = match_spans_to_docling_elements_with_reporting(docling_json, pdf_spans, overlap_threshold=0.5)
    
    # Write enriched JSON
    print(f"\nWriting enriched JSON to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_json, f, indent=2, ensure_ascii=False)
    
    print("Enrichment completed!")
    return enriched_json


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python enrich_docling_fixed.py <docling_json_path> <pdf_path> [output_path]")
        sys.exit(1)
    
    docling_json_path = sys.argv[1]
    pdf_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        enriched_json = enrich_docling_with_pymupdf(docling_json_path, pdf_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
