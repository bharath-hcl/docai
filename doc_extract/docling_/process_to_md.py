from typing import Literal, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import json
import re
import tiktoken
import hashlib

from doc_extract.models.document import (Paragraph, CodeBlock, Figure, ListBlock, CellType,
                                       TableBlock, ContentBlock, Children, Document, Segment,
                                       ChildrenMD, DocumentMD)

# Rebuild models for forward references
Children.model_rebuild()
Document.model_rebuild()
ChildrenMD.model_rebuild()
DocumentMD.model_rebuild()

def calculate_md5_hash(text: str) -> str:
    """Calculate MD5 hash of a string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_token_encoder(model_name: str = "gpt-4"):
    """Get tiktoken encoder for the specified model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, encoder: tiktoken.Encoding = None) -> int:
    """Count tokens in text using tiktoken."""
    if encoder is None:
        encoder = get_token_encoder()
    if not text or not isinstance(text, str):
        return 0
    return len(encoder.encode(text))

def create_markdown_heading(title: str, level: int) -> str:
    """Create a markdown heading based on title and level."""
    if not title:
        return ""
    heading_level = max(1, level + 1)
    heading_prefix = "#" * heading_level
    return f"{heading_prefix} {title}\n\n"

def build_provenance_path(doc_title: str, child_path: List[str] = None) -> str:
    """Build a provenance path string showing the hierarchical location."""
    path_parts = [f"doc: {doc_title or 'Untitled Document'}"]
    if child_path:
        for child_title in child_path:
            path_parts.append(f"child: {child_title or 'Untitled Section'}")
    return "/".join(path_parts)

def determine_content_label(block: ContentBlock) -> str:
    """Determine the label for a content block based on its type."""
    if block.type == "table":
        return "table"
    elif block.type == "figure":
        return "figure"
    else:
        return "text"

def recursive_character_split(text: str, token_threshold: int, encoder: tiktoken.Encoding,
                            page_no: Optional[Union[int, List[int]]] = None) -> List[dict]:
    """Recursively split text using character-based splitting to stay under token threshold."""
    if count_tokens(text, encoder) <= token_threshold:
        return [{"content": text, "token_count": count_tokens(text, encoder), "page_no": page_no}]

    split_patterns = [
        r'(?<=[.!?])\s+',  # Split on sentence boundaries
        r'\n\n',  # Split on paragraph breaks
        r'\n',  # Split on line breaks
        r'(?<=\w)\s+',  # Split on word boundaries
        r'(?<=\w)',  # Split on characters (last resort)
    ]

    for pattern in split_patterns:
        if pattern == r'(?<=\w)':
            chunk_size = max(1, len(text) // 2)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        else:
            chunks = re.split(pattern, text)
        
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if len(chunks) <= 1:
            continue
        
        result_chunks = []
        for chunk in chunks:
            if chunk.strip():
                sub_chunks = recursive_character_split(chunk.strip(), token_threshold, encoder, page_no)
                result_chunks.extend(sub_chunks)
        
        if len(result_chunks) > 1 or (len(result_chunks) == 1 and result_chunks[0]["token_count"] <= token_threshold):
            return result_chunks

    return [{"content": text, "token_count": count_tokens(text, encoder), "page_no": page_no}]

def extract_page_numbers_from_content_blocks(content_blocks: List[ContentBlock]) -> Optional[Union[int, List[int]]]:
    """Extract and aggregate page numbers from content blocks."""
    if not content_blocks:
        return None

    all_page_nums = []
    for block in content_blocks:
        if hasattr(block, 'page_no') and block.page_no is not None:
            if isinstance(block.page_no, int):
                all_page_nums.append(block.page_no)
            elif isinstance(block.page_no, list):
                all_page_nums.extend(block.page_no)

    if not all_page_nums:
        return None

    unique_pages = sorted(list(set(all_page_nums)))
    return unique_pages[0] if len(unique_pages) == 1 else unique_pages

def aggregate_page_numbers(page_nums_list: List[Optional[Union[int, List[int]]]]) -> Optional[Union[int, List[int]]]:
    """Aggregate page numbers from multiple sources."""
    all_pages = []
    for page_nums in page_nums_list:
        if page_nums is not None:
            if isinstance(page_nums, int):
                all_pages.append(page_nums)
            elif isinstance(page_nums, list):
                all_pages.extend(page_nums)

    if not all_pages:
        return None

    unique_pages = sorted(list(set(all_pages)))
    return unique_pages[0] if len(unique_pages) == 1 else unique_pages

def content_block_to_markdown(block: ContentBlock, md_text: str) -> str:
    """Convert a ContentBlock to its markdown representation."""
    if block.type == "paragraph":
        return block.text + "\n\n"
    elif block.type == "code_block":
        return f"``````\n\n"
    elif block.type == "figure":
        return f"![{block.caption}]()\n\n"
    elif block.type == "list":
        list_items = "\n".join([f"- {item}" for item in block.items])
        return f"{list_items}\n\n"
    elif block.type == "table":
        # For tables, we'll extract the actual table content from the cells field
        if hasattr(block, 'cells') and block.cells:
            # Parse the cells field to construct markdown table
            try:
                import json
                cells_data = json.loads(block.cells) if isinstance(block.cells, str) else block.cells
                
                if isinstance(cells_data, list) and cells_data:
                    # Convert cells data to markdown table
                    table_content = construct_markdown_table_from_cells(cells_data, block.dimensions)
                else:
                    # Fallback to original cells string if it's already markdown
                    table_content = block.cells if isinstance(block.cells, str) else ""
            except:
                # Fallback to original cells string
                table_content = block.cells if isinstance(block.cells, str) else ""
        else:
            table_content = "> Table content not found"
        
        caption_text = block.caption if block.caption else ""
        if caption_text:
            return f"**{caption_text}**\n\n{table_content}\n\n"
        else:
            return f"{table_content}\n\n"
    
    return ""

def construct_markdown_table_from_cells(cells_data, dimensions):
    """Construct markdown table from cells data."""
    if not cells_data or not dimensions or len(dimensions) < 2:
        return "> Table structure invalid"
    
    max_rows, max_cols = dimensions[0], dimensions[1]
    
    # Create a grid to hold cell contents
    grid = [["" for _ in range(max_cols)] for _ in range(max_rows)]
    
    # Fill the grid with cell data
    for cell in cells_data:
        if isinstance(cell, dict) and 'pos' in cell and 'text' in cell:
            pos = cell['pos']
            if len(pos) >= 2:
                row, col = pos[0], pos[1]
                if 0 <= row < max_rows and 0 <= col < max_cols:
                    grid[row][col] = cell['text']
    
    # Convert grid to markdown table
    if not any(any(row) for row in grid):
        return "> Empty table"
    
    markdown_lines = []
    for i, row in enumerate(grid):
        # Create table row
        row_content = "| " + " | ".join(row) + " |"
        markdown_lines.append(row_content)
        
        # Add separator after first row (header)
        if i == 0:
            separator = "|" + "|".join(["-" * (len(cell) + 2) for cell in row]) + "|"
            markdown_lines.append(separator)
    
    return "\n".join(markdown_lines)

def create_single_block_segment(block: ContentBlock, md_text: str, encoder: tiktoken.Encoding,
                               provenance_path: str, segment_number: int) -> Segment:
    """Create a segment for a single table or figure block."""
    block_md = content_block_to_markdown(block, md_text)
    token_count = count_tokens(block_md, encoder)
    page_no = getattr(block, 'page_no', None)
    label = determine_content_label(block)
    
    segment_id = f"segment-{calculate_md5_hash(block_md.strip())}"
    
    return Segment(
        content=block_md.strip(),
        page_no=page_no,
        token_count=token_count,
        segment_number=segment_number,
        segment_id=segment_id,
        provenance_path=provenance_path,
        label=label
    )

def create_text_segments(text_blocks: List[ContentBlock], md_text: str, encoder: tiktoken.Encoding,
                        token_threshold: int, heading_content: str, provenance_path: str,
                        start_segment_number: int) -> List[Segment]:
    """Create text segments from non-table/figure content blocks."""
    if not text_blocks and not heading_content:
        return []
    
    # Convert text blocks to markdown
    block_markdowns = []
    block_tokens = []
    block_pages = []
    
    for block in text_blocks:
        block_md = content_block_to_markdown(block, md_text)
        block_token_count = count_tokens(block_md, encoder)
        block_page_no = getattr(block, 'page_no', None)
        
        # Handle oversized blocks with recursive splitting
        if block_token_count > token_threshold:
            split_chunks = recursive_character_split(block_md.strip(), token_threshold, encoder, block_page_no)
            for chunk_data in split_chunks:
                block_markdowns.append(chunk_data["content"] + "\n\n")
                block_tokens.append(chunk_data["token_count"])
                block_pages.append(chunk_data["page_no"])
        else:
            block_markdowns.append(block_md)
            block_tokens.append(block_token_count)
            block_pages.append(block_page_no)
    
    # Calculate total tokens including heading
    total_content_tokens = sum(block_tokens)
    heading_tokens = count_tokens(heading_content, encoder)
    total_tokens = heading_tokens + total_content_tokens
    
    # If total content fits in one segment
    if total_tokens <= token_threshold:
        full_content = heading_content + "".join(block_markdowns)
        segment_id = f"segment-{calculate_md5_hash(full_content.strip())}"
        
        return [Segment(
            content=full_content.strip(),
            page_no=aggregate_page_numbers(block_pages),
            token_count=total_tokens,
            segment_number=start_segment_number,
            segment_id=segment_id,
            provenance_path=provenance_path,
            label="text"
        )]
    
    # Create multiple text segments
    segments = []
    current_segment_blocks = []
    current_segment_tokens = heading_tokens
    current_segment_pages = []
    segment_number = start_segment_number
    current_segment_content = heading_content
    
    for i, (block_md, block_token_count, block_page_no) in enumerate(zip(block_markdowns, block_tokens, block_pages)):
        if current_segment_tokens + block_token_count > token_threshold and current_segment_blocks:
            # Finish current segment
            segment_page_no = aggregate_page_numbers(current_segment_pages) if current_segment_pages else None
            segment_id = f"segment-{calculate_md5_hash(current_segment_content.strip())}"
            
            segments.append(Segment(
                content=current_segment_content.strip(),
                page_no=segment_page_no,
                token_count=current_segment_tokens,
                segment_number=segment_number,
                segment_id=segment_id,
                provenance_path=provenance_path,
                label="text"
            ))
            
            segment_number += 1
            current_segment_blocks = [i]
            current_segment_tokens = block_token_count
            current_segment_content = block_md
            current_segment_pages = [block_page_no] if block_page_no else []
        else:
            current_segment_blocks.append(i)
            current_segment_tokens += block_token_count
            current_segment_content += block_md
            if block_page_no:
                current_segment_pages.append(block_page_no)
    
    # Add final segment
    if current_segment_blocks or current_segment_content.strip():
        segment_page_no = aggregate_page_numbers(current_segment_pages) if current_segment_pages else None
        segment_id = f"segment-{calculate_md5_hash(current_segment_content.strip())}"
        
        segments.append(Segment(
            content=current_segment_content.strip(),
            page_no=segment_page_no,
            token_count=current_segment_tokens,
            segment_number=segment_number,
            segment_id=segment_id,
            provenance_path=provenance_path,
            label="text"
        ))
    
    return segments

def create_content_segments(content_blocks: List[ContentBlock], md_text: str, encoder: tiktoken.Encoding,
                          token_threshold: int = 512, title: str = None, level: int = None,
                          provenance_path: str = None) -> List[Segment]:
    """
    Create content segments with separate handling for tables and figures.
    Tables and figures get their own segments, while other content is grouped as text segments.
    """
    if not content_blocks and not title:
        return []

    if provenance_path is None:
        provenance_path = "doc: Unknown"

    segments = []
    segment_number = 1

    # Handle title/heading first if it exists
    heading_content = ""
    if title and level is not None:
        heading_content = create_markdown_heading(title, level)

    # Separate content blocks by type
    text_blocks = []
    
    for block in content_blocks:
        if block.type in ["table", "figure"]:
            # If we have accumulated text blocks, process them first
            if text_blocks:
                text_segments = create_text_segments(
                    text_blocks, md_text, encoder, token_threshold, 
                    heading_content, provenance_path, segment_number
                )
                segments.extend(text_segments)
                segment_number += len(text_segments)
                text_blocks = []
                heading_content = ""  # Only include heading in first segment
            
            # Process table/figure as separate segment
            table_figure_segment = create_single_block_segment(
                block, md_text, encoder, provenance_path, segment_number
            )
            segments.append(table_figure_segment)
            segment_number += 1
        else:
            text_blocks.append(block)
    
    # Process any remaining text blocks
    if text_blocks or heading_content:
        text_segments = create_text_segments(
            text_blocks, md_text, encoder, token_threshold,
            heading_content, provenance_path, segment_number
        )
        segments.extend(text_segments)

    return segments if segments else []

def calculate_children_token_counts_and_child_ids(children_md: List[ChildrenMD], encoder: tiktoken.Encoding) -> tuple:
    """Calculate token counts and child IDs for children recursively and return total count and combined hash."""
    total_tokens = 0
    combined_hash_strings = []

    for child in children_md:
        # Calculate token count and child ID for this child's content
        child_content_tokens = sum(segment.token_count or 0 for segment in child.content)
        
        # Concatenate all segment hashes and create a new hash
        if child.content:
            segment_hashes = [segment.segment_id.replace("segment-", "") for segment in child.content]
            content_hash = calculate_md5_hash("".join(segment_hashes))
        else:
            content_hash = calculate_md5_hash("")

        # Set child_id for this child
        child.child_id = f"child-{content_hash}"
        combined_hash_strings.append(content_hash)

        # Recursively calculate token counts and child IDs for sub-children
        children_tokens, children_hash_strings = calculate_children_token_counts_and_child_ids(child.children, encoder)
        combined_hash_strings.extend(children_hash_strings)

        # Set this child's total token count (content + all children)
        child.token_count = child_content_tokens + children_tokens

        # Add to total
        total_tokens += child.token_count

    return total_tokens, combined_hash_strings

def process_children_to_md(children: List[Children], md_text: str, encoder: tiktoken.Encoding,
                          token_threshold: int = 512, doc_title: str = None,
                          parent_path: List[str] = None) -> List[ChildrenMD]:
    """Recursively process Children objects into ChildrenMD objects."""
    children_md = []
    if parent_path is None:
        parent_path = []

    for child in children:
        # Build provenance path for this child
        current_path = parent_path + ([child.title] if child.title else ["Untitled Section"])
        provenance_path = build_provenance_path(doc_title, current_path)

        # Create content segments based on token threshold
        content_segments = create_content_segments(
            child.content, md_text, encoder, token_threshold, child.title, child.level, provenance_path
        )

        # Extract page numbers from content blocks
        content_page_nums = extract_page_numbers_from_content_blocks(child.content)

        # Recursively process child sections with updated path
        sub_children_md = process_children_to_md(child.children, md_text, encoder, token_threshold, doc_title, current_path)

        # Aggregate page numbers from sub-children
        sub_children_page_nums = [sub_child.page_no for sub_child in sub_children_md]
        aggregated_page_nums = aggregate_page_numbers([content_page_nums] + sub_children_page_nums)

        # Create ChildrenMD object with page numbers
        child_md = ChildrenMD(
            title=child.title,
            level=child.level,
            content=content_segments,
            child_id="",  # Will be set in calculate_children_token_counts_and_child_ids
            children=sub_children_md,
            token_count=None,  # Will be calculated later
            page_no=aggregated_page_nums
        )

        children_md.append(child_md)

    return children_md

def document_to_documentmd(doc: Document, md_text: str, token_threshold: int = 512) -> DocumentMD:
    """Transform a Document object into a DocumentMD object."""
    # Initialize token encoder
    encoder = get_token_encoder()

    # Build root provenance path
    doc_title = doc.title or "Untitled Document"
    root_provenance_path = build_provenance_path(doc_title)

    # Create content segments for root-level content
    root_content_segments = create_content_segments(
        doc.content, md_text, encoder, token_threshold, doc.title, doc.level, root_provenance_path
    )

    # Process children recursively
    children_md = process_children_to_md(doc.children, md_text, encoder, token_threshold, doc_title)

    # Calculate token counts and child IDs for all children
    children_total_tokens, children_hash_strings = calculate_children_token_counts_and_child_ids(children_md, encoder)

    # Calculate token count and hash for root content
    root_content_tokens = sum(segment.token_count or 0 for segment in root_content_segments)

    # Calculate root content hash by concatenating all segment hashes
    if root_content_segments:
        segment_hashes = [segment.segment_id.replace("segment-", "") for segment in root_content_segments]
        root_content_hash = calculate_md5_hash("".join(segment_hashes))
    else:
        root_content_hash = calculate_md5_hash("")

    # Calculate document ID by combining root content hash and all children hashes
    all_hash_strings = [root_content_hash] + children_hash_strings
    combined_hash = calculate_md5_hash("".join(all_hash_strings))
    doc_id = f"document-{combined_hash}"

    # Extract page numbers from root-level content
    root_page_nums = extract_page_numbers_from_content_blocks(doc.content)

    # Aggregate page numbers from children
    children_page_nums = [child.page_no for child in children_md]
    all_page_nums = aggregate_page_numbers([root_page_nums] + children_page_nums)

    # Create DocumentMD object
    doc_md = DocumentMD(
        file_name=doc.file_name,
        title=doc.title,
        level=doc.level,
        content=root_content_segments,
        children=children_md,
        token_count=root_content_tokens + children_total_tokens,
        doc_id=doc_id
    )

    return doc_md

def load_document_from_json(json_file_path: str) -> Document:
    """Load Document object from JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Document(**data)

def load_markdown_from_file(md_file_path: str) -> str:
    """Load markdown content from file."""
    with open(md_file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_documentmd_to_json(doc_md: DocumentMD, output_path: str):
    """Save DocumentMD object to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(doc_md.model_dump(), f, indent=2, ensure_ascii=False)

def print_token_summary(doc_md: DocumentMD, indent: int = 0):
    """Print a summary of token counts and segmentation info."""
    prefix = " " * indent
    title = doc_md.title if doc_md.title else "Root Document"
    token_count = doc_md.token_count if doc_md.token_count is not None else 0
    
    # Count segments by label
    label_counts = {"text": 0, "table": 0, "figure": 0}
    for segment in doc_md.content:
        label_counts[segment.label] += 1
    
    content_info = f" ({label_counts['text']} text, {label_counts['table']} tables, {label_counts['figure']} figures)"
    
    print(f"{prefix}{title}: {token_count:,} tokens{content_info} [ID: {doc_md.doc_id}]")
    
    for child in doc_md.children:
        print_child_token_summary(child, indent + 1)

def print_child_token_summary(child_md: ChildrenMD, indent: int = 0):
    """Print token summary for a child section."""
    prefix = " " * indent
    title = child_md.title if child_md.title else "Untitled Section"
    token_count = child_md.token_count if child_md.token_count is not None else 0
    
    # Count segments by label
    label_counts = {"text": 0, "table": 0, "figure": 0}
    for segment in child_md.content:
        label_counts[segment.label] += 1
    
    content_info = f" ({label_counts['text']} text, {label_counts['table']} tables, {label_counts['figure']} figures)"
    
    print(f"{prefix}{title}: {token_count:,} tokens{content_info} [ID: {child_md.child_id}]")
    
    for sub_child in child_md.children:
        print_child_token_summary(sub_child, indent + 1)

def main():
    """Main function to demonstrate the transformation process."""
    # Example usage:
    json_file = "docling_final.json"  # Your input file
    md_file = "docling.md"  # Your markdown file
    output_file = "docling_documentmd_with_labels.json"

    # Token threshold for segmentation
    token_threshold = 512

    print(f"Loading files...")
    
    # Load the Document object
    document = load_document_from_json(json_file)
    
    # Load the markdown content
    markdown_content = load_markdown_from_file(md_file)

    print(f"Transforming to DocumentMD with labels and separate table/figure handling...")
    print(f"Token threshold: {token_threshold}")
    print("Features:")
    print("- Tables and figures get separate segments with appropriate labels")
    print("- Text content grouped into text segments")
    print("- No duplicate content")
    print("- Segments: segment_id with 'segment-{md5_hash}' format")
    print("- Children: child_id with 'child-{md5_hash}' format")
    print("- Documents: doc_id with 'document-{md5_hash}' format")

    # Transform to DocumentMD
    document_md = document_to_documentmd(document, markdown_content, token_threshold)

    # Save the result
    save_documentmd_to_json(document_md, output_file)

    print(f"Transformation complete! Output saved to {output_file}")
    print(f"\nToken Count and Segmentation Summary:")
    print("=" * 50)
    print_token_summary(document_md)
    print("=" * 50)
    total_tokens = document_md.token_count if document_md.token_count is not None else 0
    print(f"Total document tokens: {total_tokens:,}")
    print(f"Document ID: {document_md.doc_id}")

    print("\nScript completed successfully!")
    print("\nKey improvements:")
    print("- Added 'label' field to distinguish text, table, and figure content")
    print("- Tables and figures are processed as separate segments")
    print("- No duplicate table content")
    print("- Proper markdown table construction from cells data")

if __name__ == "__main__":
    main()
