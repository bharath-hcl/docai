#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Docling → compact-JSON converter
 • works for ANY DoclingDocument (paper, bill, story …)
 • keeps page numbers
 • keeps logical hierarchy (sections / sub-sections)
 • tables → flat list of cell-dicts (bbox ignored)
 • ZERO hard-wired knowledge about “Front-Matter” or “Abstract”
"""

import json, re
from typing import Any, Dict, List, Optional


###########################
# small helpers
###########################

def _page_no_from_prov(prov):
    if not isinstance(prov, list):
        return None
    pages = [p.get("page_no") for p in prov if isinstance(p, dict)]
    pages = [p for p in pages if isinstance(p, int)]
    return min(pages) if pages else None


def _ref(o):
    return o.get("$ref") or o.get("cref") if isinstance(o, dict) else None


###########################
# converter
###########################

class DoclingToRagConverter:
    def __init__(self, js: Dict[str, Any]):
        self.src = js
        self.ref: Dict[str, Dict[str, Any]] = {}
        self.out: Dict[str, Any] = {}

    # ---------- public API
    def convert(self) -> Dict[str, Any]:
        self._build_ref_map()
        self._walk_body()
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
                        (self.ref[r]["page_no"] for r in li if self.ref[r]["page_no"] is not None),
                        default=None,
                    ),
                }
            else:
                txt = " ".join(
                    self.ref.get(_ref(ch), {}).get("text", "") for ch in g.get("children", [])
                ).strip()
                self.ref[sid] = {
                    "kind": "text",
                    "text": txt,
                    "page_no": _page_no_from_prov(g.get("prov")),
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
            out.append(
                {
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
                }
            )
        return out

    # ---------- pass 2 : walk body ---------------------------------------------
    def _walk_body(self):
        self.out = {
            "document_id": self.src.get("name"),
            "title": "",
            "sections": [],
        }

        stack: List[Dict[str, Any]] = []

        def cur_sec():
            if stack:
                return stack[-1]
            root = {"title": None, "level": 0, "content": [], "sub_sections": []}
            self.out["sections"].append(root)
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
                sec = {"title": it["text"], "level": lvl, "content": [], "sub_sections": []}
                (stack[-1] if stack else self.out)["sections" if not stack else "sub_sections"].append(sec)
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


###########################
# CLI helper
###########################

if __name__ == "__main__":
    import argparse, pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("docling_json")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    inp = pathlib.Path(args.docling_json)
    out = pathlib.Path(args.out or inp.with_suffix("").as_posix() + "_rag.json")

    with inp.open("r", encoding="utf-8") as f:
        js = json.load(f)

    conv = DoclingToRagConverter(js)
    out.write_text(json.dumps(conv.convert(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓  converted → {out}")
