import re

from typing import Any, Dict, List, Optional

###########################
# small helpers
###########################

def _page_no_from_prov(prov):
    if not isinstance(prov, list):
        return None
    pages = [p.get("page_no") for p in prov if isinstance(p, dict)]
    pages = [p for p in pages if isinstance(p, int)]
    return [min(pages)] if pages else None

def _ref(o):
    return o.get("$ref") or o.get("cref") if isinstance(o, dict) else None

###########################
# Post Processor
###########################

def process(doc_dict):
    doc_dict_processor = DoclingPostProcessor(doc_dict)
    return doc_dict_processor.convert()

class DoclingPostProcessor:
    def __init__(self, js: Dict[str, Any]):
        self.src = js
        self.ref: Dict[str, Dict[str, Any]] = {}
        self.out: Dict[str, Any] = {}

    # ---------- public API

    def convert(self) -> Dict[str, Any]:
        self._build_ref_map()
        self._walk_body()
        self._merge_split_paragraphs()
        self._remove_empty_children()  # Add cleanup for empty children
        return self.out

    # ---------- pass 1 : build miniature objects --------------------------------

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

        # ------------ tables ----------------------------------------------------
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

            self.ref[sid] = {
                "kind": "table",
                "caption": caption,
                "cells": self._extract_cells_flat(tbl),
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

                self.ref[sid] = {
                    "kind": "list",
                    "items": [self.ref[r]["text"] for r in li if self.ref[r]["text"]],
                    "page_no": min(
                        (self.ref[r]["page_no"] for r in li if self.ref[r].get("page_no") is not None),
                        default=None,
                    ),
                }
            else:
                child_refs = [_ref(ch) for ch in g.get("children", [])]
                child_refs = [r for r in child_refs if r and self.ref.get(r)]
                
                txt = " ".join(
                    self.ref.get(r, {}).get("text", "") for r in child_refs
                ).strip()
                
                # Extract page numbers from children
                child_page_nos = []
                for r in child_refs:
                    if r in self.ref and self.ref[r].get("page_no"):
                        child_page_nos.extend(self.ref[r]["page_no"])
                
                page_no = [min(child_page_nos)] if child_page_nos else None
                
                self.ref[sid] = {
                    "kind": "text",
                    "text": txt,
                    "page_no": page_no,
                }

    # ---------- table-cell helper ----------------------------------------------

    def _extract_cells_flat(self, tbl) -> List[Dict[str, Any]]:
        data = tbl.get("data", {}) or tbl.get("structure", {})
        src = data.get("table_cells") or data.get("cells") or []
        out = []

        for c in src:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue

            r0 = (c.get("row_start") or 0)
            c0 = (c.get("col_start") or 0)
            rs = (c.get("row_span") or 1)
            cs = (c.get("col_span") or 1)

            out.append({
                "row_span": rs,
                "col_span": cs,
                "start_row_offset_idx": r0,
                "end_row_offset_idx": r0 + rs,
                "start_col_offset_idx": c0,
                "end_col_offset_idx": c0 + cs,
                "text": txt,
                "column_header": bool(c.get("is_col_header")),
                "row_header": bool(c.get("is_row_header")),
                "row_section": bool(c.get("is_row_header") and c0 == 0),
            })

        return out

    # ---------- paragraph merging logic -------------------------------------

    def _merge_split_paragraphs(self):
        """Merge paragraphs that are split across pages"""
        
        def is_sentence_ending(text):
            """Check if text ends with sentence-ending punctuation"""
            if not text:
                return True
            # Check for common sentence endings
            return text.rstrip()[-1:] in '.!?;:'
        
        def merge_in_content(content_list):
            i = 0
            while i < len(content_list):
                current = content_list[i]
                
                # Only process paragraphs
                if current.get("type") != "paragraph":
                    i += 1
                    continue
                
                current_text = current.get("text", "").strip()
                current_page_no = current.get("page_no")
                
                # Check if current paragraph doesn't end with sentence-ending punctuation
                if is_sentence_ending(current_text):
                    i += 1
                    continue
                
                # Look for the next paragraph (skip figures, tables, etc.)
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
                
                # Check if page numbers are valid and consecutive
                if (current_page_no and next_page_no and 
                    isinstance(current_page_no, list) and isinstance(next_page_no, list) and 
                    len(current_page_no) > 0 and len(next_page_no) > 0 and
                    next_page_no[0] == current_page_no[0] + 1):
                    
                    # Merge the paragraphs
                    merged_text = current_text + " " + next_para.get("text", "").strip()
                    merged_page_no = list(set(current_page_no + next_page_no))  # Remove duplicates
                    merged_page_no.sort()
                    
                    # Update current paragraph
                    current["text"] = merged_text
                    current["page_no"] = merged_page_no
                    
                    # Remove the next paragraph
                    content_list.pop(next_para_idx)
                    
                    # Don't increment i, as we might need to merge with another paragraph
                    continue
                
                i += 1
        
        def merge_in_sections(sections):
            for section in sections:
                # Merge paragraphs in current section's content
                merge_in_content(section["content"])
                
                # Recursively handle child sections
                if section.get("children"):
                    merge_in_sections(section["children"])
        
        # Start merging from the root children
        merge_in_sections(self.out["children"])

    # ---------- remove empty children arrays -------------------------------

    def _remove_empty_children(self):
        """Remove empty children arrays from all sections"""
        
        def clean_section(section):
            # Recursively clean child sections first
            if "children" in section and section["children"]:
                for child in section["children"]:
                    clean_section(child)
                # After cleaning children, check if any are left
                if not section["children"]:
                    del section["children"]
            elif "children" in section and not section["children"]:
                # Remove empty children array
                del section["children"]
        
        # Clean all top-level sections
        for section in self.out["children"]:
            clean_section(section)

    # ---------- pass 2 : walk body ---------------------------------------------

    def _walk_body(self):
        self.out = {
            "document_id": self.src.get("name"),
            "title": "",
            "children": [],
        }

        stack: List[Dict[str, Any]] = []

        def cur_sec():
            if stack:
                return stack[-1]
            # Don't include children array initially
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

            # ------------ document title
            if kind == "title":
                self.out["title"] = it["text"]
                continue

            # ------------ section header
            if kind == "section_header":
                lvl = it.get("level") or 1

                while stack and stack[-1]["level"] >= lvl:
                    stack.pop()

                # Don't include children array initially
                sec = {"title": it["text"], "level": lvl, "content": []}
                
                parent = stack[-1] if stack else self.out
                # Only create children array when we actually need to add something
                if "children" not in parent:
                    parent["children"] = []
                parent["children"].append(sec)
                stack.append(sec)
                continue

            # ------------ normal content
            payload = None

            if kind in {"text", "paragraph"} and it["text"]:
                payload = {"type": "paragraph", "text": it["text"], "page_no": it["page_no"]}
            elif kind == "code" and it["text"]:
                payload = {"type": "code_block", "text": it["text"], "page_no": it["page_no"]}
            elif kind == "figure" and it["caption"]:
                payload = {"type": "figure", "caption": it["caption"], "page_no": it["page_no"]}
            elif kind == "list" and it["items"]:
                payload = {"type": "list", "items": it["items"], "page_no": it["page_no"]}
            elif kind == "table" and (it["cells"] or it["caption"]):
                payload = {
                    "type": "table",
                    "caption": it["caption"],
                    "cells": it["cells"],
                    "cell_count": len(it["cells"]),
                    "page_no": it["page_no"],
                }

            if payload:
                cur_sec()["content"].append(payload)
