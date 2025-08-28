import re
from typing import Any, Dict, List, Optional, Union
import copy
import numpy as np
import json
# Import the Pydantic models
from doc_extract.models.document import (Document, Children, ContentBlock, Paragraph, CellType,
                                         CodeBlock, Figure, ListBlock, TableBlock, TableCell
                                         )

###########################
# small helpers 
###########################

def _page_no_from_prov(prov):
    if not isinstance(prov, list):
        return None
    pages = [p.get("page_no") for p in prov if isinstance(p, dict)]
    pages = [p for p in pages if isinstance(p, int)]
    return min(pages) if pages else None # Return single int

def _ref(o):
    return o.get("$ref") or o.get("cref") if isinstance(o, dict) else None

import tiktoken

def get_token_count(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        if not text:
            return 0
        return len(encoding.encode(text))
    except ImportError:
        if not text:
            return 0
        # OPT: use ~0.75 tokens per word for OpenAI models
        return int(len(text.split()) * 0.75)
    
def _create_simple_format(cells_data: List[Dict], max_row: int, max_col: int) -> Dict[str, Any]:
    """Create simple row-based format"""
    # Create grid
    grid = [[None for _ in range(max_col)] for _ in range(max_row)]
    
    # Fill grid with cells
    for cell in cells_data:
        r, c = cell["pos"]
        rs, cs = cell.get("span", [1, 1])
        
        # Fill all spanned positions
        for ri in range(r, r + rs):
            for ci in range(c, c + cs):
                if ri < max_row and ci < max_col:
                    grid[ri][ci] = cell
    
    # Convert to rows
    rows = []
    for r in range(max_row):
        row_cells = []
        row_type = "data"
        
        for c in range(max_col):
            cell = grid[r][c]
            if cell:
                row_cells.append(cell["text"])
                if cell["type"] in ["col_header", "row_header"]:
                    row_type = "header"
            else:
                row_cells.append("")
        
        # Only include non-empty rows
        if any(cell.strip() for cell in row_cells):
            rows.append({"cells": row_cells, "type": row_type})
    
    return {"simple": True, "rows": rows}

def _create_complex_format(cells_data: List[Dict], max_row: int, max_col: int) -> Dict[str, Any]:
    """Create complex cell-based format for tables with spans"""
    return {
        "simple": False,
        "cells": cells_data,
        "dimensions": [max_row, max_col]
    }


###########################
# Post Processor 
###########################

def process(doc_dict) -> Document:
    """
    Process docling JSON output and return a Document Pydantic model
    """
    doc_dict_processor = DoclingPostProcessor(doc_dict)
    return doc_dict_processor.convert()

class DoclingPostProcessor:
    def __init__(self, js: Dict[str, Any]):
        self.src = js
        self.ref: Dict[str, Dict[str, Any]] = {}
        self.out: Dict[str, Any] = {}

    # ---------- public API
    def convert(self) -> Document:
        """Convert to Pydantic Document model"""
        self._build_ref_map()
        self._walk_body()
        self._merge_split_paragraphs()
        self._remove_empty_children()
        
        # Convert dictionary output to Pydantic models
        return self._convert_to_document_model()

    def _convert_to_document_model(self) -> Document:
        """Convert the dictionary output to Document Pydantic model"""
        children = [self._convert_section_to_children(child) for child in self.out.get("children", [])]
        if len(children)==1:
                return Document(
                file_name=self.out.get("file_name"),
                title=children[0].title,
                content = children[0].content,
                children = children[0].children
                )
        else:
            return Document(
                file_name=self.out.get("file_name"),
                title=self.out.get("title"),
                children=children
            )

    def _convert_section_to_children(self, section: Dict[str, Any]) -> Children:
        """Convert a section dictionary to Children Pydantic model"""
        content_models = [self._convert_content_to_model(c) for c in section.get("content", [])]
        children_models = [self._convert_section_to_children(c) for c in section.get("children", [])]
        
        return Children(
            title=section.get("title"),
            level=section.get("level", None),
            content=content_models,
            children=children_models
        )
    
    def _convert_content_to_model(self, content: Dict[str, Any]) -> ContentBlock:
        content_type = content["type"]

        if content_type == "paragraph":
            txt = content["text"]
            return Paragraph(
                text=txt,
                page_no=content.get("page_no"),
                token_count=get_token_count(txt)
            )
        elif content_type == "code_block":
            txt = content["text"]
            return CodeBlock(
                text=txt,
                page_no=content.get("page_no"),
                token_count=get_token_count(txt)
            )
        elif content_type == "figure":
            caption = content["caption"]
            return Figure(
                caption=caption,
                page_no=content.get("page_no"),
                token_count=get_token_count(caption)
            )
        elif content_type == "list":
            items = content["items"]
            return ListBlock(
                items=items,
                page_no=content.get("page_no"),
                token_count=sum(get_token_count(item) for item in items)
            )
        elif content_type == "table":
            return self._convert_table_to_model(content)
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    def _convert_table_to_model(self, content: Dict[str, Any]) -> TableBlock:
        """COMPLETELY REWRITTEN: Convert table content with bulletproof indexing"""
        caption = content.get("caption")
        page_no = content.get("page_no")
        
        # Check if we have optimized table data
        optimized_table = content.get("optimized_table")
        if not optimized_table:
            # Legacy format - direct conversion
            return self._convert_legacy_table(content, caption, page_no)
        # Handle optimized table format
        if optimized_table.get("simple"):
            return self._convert_simple_table(optimized_table, caption, page_no)
        else:
            return self._convert_complex_table(optimized_table, caption, page_no)

    def _convert_legacy_table(self, content: Dict[str, Any], caption: str, page_no) -> TableBlock:
        """Convert legacy format tables"""
        cells = [
            TableCell(
                text=cell["text"],
                row_span=cell["row_span"],
                col_span=cell["col_span"],
                start_row_offset_idx=cell["start_row_offset_idx"],
                end_row_offset_idx=cell["end_row_offset_idx"],
                start_col_offset_idx=cell["start_col_offset_idx"],
                end_col_offset_idx=cell["end_col_offset_idx"],
                column_header=cell["column_header"],
                row_header=cell["row_header"],
                row_section=cell["row_section"]
            ) for cell in content.get("cells", [])
        ]
        
        return TableBlock(
            caption=caption,
            cells=cells,
            cell_count=content.get("cell_count", 0),
            page_no=page_no
        )

    def _convert_simple_table(self, table_data: Dict[str, Any], caption: str, page_no) -> TableBlock:
        """Convert simple table format"""
        cells_list = []
        
        for row_idx, row in enumerate(table_data["rows"]):
            for col_idx, cell_text in enumerate(row["cells"]):
                # ✅ DEFENSIVE: Handle cell type safely
                cell_type_str = row.get("type", "data")
                if cell_type_str == "header":
                    cell_type = CellType.HEADER
                else:
                    cell_type = CellType.DATA
                
                cells_list.append(TableCell(
                    text=cell_text,
                    pos=[row_idx, col_idx],  # [row, col] position
                    type=cell_type,
                    span=None  # Simple tables don't have spans
                ).dict())

        # Calculate dimensions
        max_rows = len(table_data["rows"]) if table_data.get("rows") else 0
        max_cols = max(len(row["cells"]) for row in table_data["rows"]) if table_data.get("rows") else 0

        return TableBlock(
            type="table",
            caption=caption,
            cells=json.dumps(cells_list),
            dimensions=[max_rows, max_cols],
            page_no=page_no,
            token_count=get_token_count(json.dumps(cells_list))
        )

    def _convert_complex_table(self, table_data: Dict[str, Any], caption: str, page_no) -> 'TableBlock':
        """Convert complex table format with DEFENSIVE PROGRAMMING"""
        cells_list = []
        
        for cell in table_data["cells"]:
            # ✅ DEFENSIVE: Extract position coordinates explicitly
            cell_position = cell.get("pos", [0, 0])
            if not isinstance(cell_position, list) or len(cell_position) < 2:
                cell_position = [0, 0]  # Fallback for malformed data
            
            # ✅ DEFENSIVE: Extract span information explicitly
            cell_span = cell.get("span", None)
            if cell_span is not None:
                if not isinstance(cell_span, list) or len(cell_span) < 2:
                    cell_span = None  # Invalid span, set to None
                else:
                    # Ensure span values are at least 1
                    cell_span = [max(1, int(cell_span[0])), max(1, int(cell_span[1]))]
            
            # ✅ DEFENSIVE: Handle cell type safely
            cell_type_str = cell.get("type", "data")
            try:
                cell_type = CellType(cell_type_str)
            except ValueError:
                cell_type = CellType.DATA  # Default fallback
            
            # Create TableCell with correct field names
            cells_list.append(TableCell(
                text=cell.get("text", ""),
                pos=cell_position,  # [row, col] position
                type=cell_type,
                span=cell_span  # [row_span, col_span] or None
            ).dict())

        # ✅ DEFENSIVE: Extract dimensions safely
        dimensions = table_data.get("dimensions", [0, 0])
        if not isinstance(dimensions, list) or len(dimensions) < 2:
            # Calculate dimensions from cell positions if not provided
            max_row = max((cell.pos[0] + (cell.span[0] if cell.span else 1)) for cell in cells_list) if cells_list else 1
            max_col = max((cell.pos[1] + (cell.span[1] if cell.span else 1)) for cell in cells_list) if cells_list else 1
            dimensions = [max_row, max_col]

        return TableBlock(
            type="table",  # Explicit type field
            caption=caption,
            cells=json.dumps(cells_list),  # Use correct field name
            dimensions=dimensions,  # Required field
            page_no=page_no,
            token_count=get_token_count(json.dumps(cells_list))  # Calculate token count
        )

    # ---------- build reference map --------------------------------
    def _build_ref_map(self):
        # ------------ texts -----------------------------------------------------
        for t in self.src.get("texts", []):
            sid = t.get("self_ref")
            if not sid:
                continue
            
            text = t.get("text") or ""
            if not text and isinstance(t.get("prov"), list):
                text = " ".join(str(p.get("text") or "") for p in t["prov"])
            text = text.strip()
            
            if t.get("label") == "list_item":
                text = re.sub(r"^\s*(?:[-•·]|\d+[.)])\s*", "", text)
            
            self.ref[sid] = {
                "kind": t.get("label", "text"),
                "text": text,
                "level": t.get("level"),
                "page_no": _page_no_from_prov(t.get("prov")),
                "content_layer": t.get("content_layer"),
            }

        # ------------ tables with optimized extraction ------------------------
        for tbl in self.src.get("tables", []):
            sid = tbl.get("self_ref")
            if not sid:
                continue
            
            caption = ""
            for ch in tbl.get("children", []):
                rid = _ref(ch)
                if rid and self.ref.get(rid, {}).get("kind") == "caption":
                    caption = self.ref[rid]["text"]
                    break
            
            # Use optimized table extraction
            optimized_table = self._extract_cells_optimized(tbl)
            
            self.ref[sid] = {
                "kind": "table",
                "caption": caption,
                "optimized_table": optimized_table,
                "page_no": _page_no_from_prov(tbl.get("prov")),
            }

        # ------------ pictures → figure ----------------------------------------
        for pic in self.src.get("pictures", []):
            sid = pic.get("self_ref")
            if not sid:
                continue
            
            caption = ""
            for ch in pic.get("children", []):
                rid = _ref(ch)
                if rid and self.ref.get(rid, {}).get("kind") == "caption":
                    caption = self.ref[rid]["text"]
                    break
            
            self.ref[sid] = {
                "kind": "figure",
                "caption": caption,
                "page_no": _page_no_from_prov(pic.get("prov")),
            }

        # ------------ groups ----------------------------------------------------
        for g in self.src.get("groups", []):
            sid = g.get("self_ref")
            if not sid:
                continue
            
            if g.get("label") == "list":
                li = [_ref(ch) for ch in g.get("children", [])]
                li = [r for r in li if r and self.ref.get(r)]
                
                # Get page numbers from list items (single integers)
                page_nos = []
                for r in li:
                    if self.ref[r].get("page_no") is not None:
                        page_nos.append(self.ref[r]["page_no"])
                page_no = min(page_nos) if page_nos else None
                
                self.ref[sid] = {
                    "kind": "list",
                    "items": [self.ref[r]["text"] for r in li if self.ref[r]["text"]],
                    "page_no": page_no,
                }
            else:
                child_refs = [_ref(ch) for ch in g.get("children", [])]
                child_refs = [r for r in child_refs if r and self.ref.get(r)]
                txt = " ".join(
                    self.ref.get(r, {}).get("text", "") for r in child_refs
                ).strip()
                
                # Extract page numbers from children (single integers)
                child_page_nos = []
                for r in child_refs:
                    if r in self.ref and self.ref[r].get("page_no") is not None:
                        child_page_nos.append(self.ref[r]["page_no"])
                page_no = min(child_page_nos) if child_page_nos else None
                
                self.ref[sid] = {
                    "kind": "text",
                    "text": txt,
                    "page_no": page_no,
                }

    def _extract_cells_optimized(self, tbl) -> Dict[str, Any]:
        """Extract table cells from Docling format and create compact representation"""
        data = tbl.get("data", {})
        src = data.get("table_cells", [])
        
        if not src:
            return {"simple": True, "rows": []}
        
        cells_data = []
        has_spans = False
        max_row, max_col = 0, 0
        
        for c in src:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            
            # Correct field names for Docling format
            r0 = c.get("start_row_offset_idx", 0)
            c0 = c.get("start_col_offset_idx", 0)
            rs = c.get("row_span", 1)
            cs = c.get("col_span", 1)
            
            max_row = max(max_row, r0 + rs)
            max_col = max(max_col, c0 + cs)
            
            # Determine cell type based on Docling format
            cell_type = "data"
            if c.get("column_header", False):
                cell_type = "col_header"
            elif c.get("row_header", False):
                cell_type = "row_header"
            
            cell = {
                "text": txt,
                "pos": [r0, c0],
                "type": cell_type
            }
            
            # Only add span if not 1,1
            if rs > 1 or cs > 1:
                cell["span"] = [rs, cs]
                has_spans = True
                
            cells_data.append(cell)
        
        # Decide if table is simple (no spans and reasonable width)
        is_simple = not has_spans and max_col <= 6
        
        if is_simple:
            return _create_simple_format(cells_data, max_row, max_col)
        else:
            return _create_complex_format(cells_data, max_row, max_col)


    # ---------- paragraph merging logic -----------------------------
    def _merge_split_paragraphs(self):
        """Merge paragraphs that are split across pages - handles both int and list page numbers"""
        def is_sentence_ending(text):
            if not text:
                return True
            return text.rstrip()[-1:] in '.!?;:'
        
        def merge_in_content(content_list):
            i = 0
            while i < len(content_list):
                current = content_list[i]
                if current.get("type") != "paragraph":
                    i += 1
                    continue
                
                current_text = current.get("text", "").strip()
                current_page_no = current.get("page_no")
                
                if is_sentence_ending(current_text):
                    i += 1
                    continue
                
                next_para_idx = None
                for j in range(i + 1, len(content_list)):
                    if content_list[j].get("type") == "paragraph":
                        next_para_idx = j
                        break
                
                if next_para_idx is None:
                    i += 1
                    continue
                
                next_para = content_list[next_para_idx]
                next_page_no = next_para.get("page_no")
                
                # Handle both int and list page numbers gracefully
                if current_page_no is not None and next_page_no is not None:
                    # Convert to lists for consistent handling
                    current_pages = current_page_no if isinstance(current_page_no, list) else [current_page_no]
                    next_pages = next_page_no if isinstance(next_page_no, list) else [next_page_no]
                    
                    # Check if consecutive pages
                    if (len(current_pages) > 0 and len(next_pages) > 0 and 
                        next_pages[0] == current_pages[-1] + 1):
                        
                        merged_text = current_text + " " + next_para.get("text", "").strip()
                        # Merge page numbers and sort
                        merged_pages = sorted(set(current_pages + next_pages))
                        
                        current["text"] = merged_text
                        current["page_no"] = merged_pages
                        content_list.pop(next_para_idx)
                        continue
                
                i += 1
        
        def merge_in_sections(sections):
            for section in sections:
                merge_in_content(section["content"])
                if section.get("children"):
                    merge_in_sections(section["children"])
        
        merge_in_sections(self.out["children"])

    def _remove_empty_children(self):
        """Remove empty children arrays from all sections"""
        def clean_section(section):
            if "children" in section and section["children"]:
                for child in section["children"]:
                    clean_section(child)
                if not section["children"]:
                    del section["children"]
            elif "children" in section and not section["children"]:
                del section["children"]
        
        for section in self.out["children"]:
            clean_section(section)

    def _walk_body(self):
        self.out = {
            "file_name": self.src.get("name"),
            "title": "",
            "children": [],
        }
        
        stack: List[Dict[str, Any]] = []
        
        def cur_sec():
            if stack:
                return stack[-1]
            root = {"title": None, "level": 0, "content": []}
            self.out["children"].append(root)
            stack.append(root)
            return root
        
        for ch in self.src.get("body", {}).get("children", []):
            rid = _ref(ch)
            it = self.ref.get(rid)
            if not it or it.get("content_layer") == "furniture":
                continue
            
            kind = it["kind"]
            
            if kind == "title":
                self.out["title"] = it["text"]
                continue
            
            if kind == "section_header":
                lvl = it.get("level", 1)
                while stack and stack[-1]["level"] >= lvl:
                    stack.pop()
                
                sec = {"title": it["text"], "level": lvl, "content": []}
                parent = stack[-1] if stack else self.out
                if "children" not in parent:
                    parent["children"] = []
                parent["children"].append(sec)
                stack.append(sec)
                continue
            
            payload = None
            if kind in {"text", "paragraph"} and it["text"]:
                payload = {"type": "paragraph", "text": it["text"], "page_no": it["page_no"]}
            elif kind == "code" and it["text"]:
                payload = {"type": "code_block", "text": it["text"], "page_no": it["page_no"]}
            elif kind == "figure" and it["caption"]:
                payload = {"type": "figure", "caption": it["caption"], "page_no": it["page_no"]}
            elif kind == "list" and it["items"]:
                payload = {"type": "list", "items": it["items"], "page_no": it["page_no"]}
            elif kind == "table" and (it.get("optimized_table") or it.get("caption")):
                payload = {
                    "type": "table",
                    "caption": it.get("caption"),
                    "optimized_table": it.get("optimized_table"),
                    "page_no": it.get("page_no"),
                }
            
            if payload:
                cur_sec()["content"].append(payload)

