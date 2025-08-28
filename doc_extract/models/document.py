from typing import Literal, List, Optional, Union, Dict
from pydantic import BaseModel, Field
from enum import Enum

class Paragraph(BaseModel):
    type: Literal["paragraph"] = Field(default="paragraph")
    text: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: int = None

class CodeBlock(BaseModel):
    type: Literal["code_block"] = Field(default="code_block")
    text: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: int = None

class Figure(BaseModel):
    type: Literal["figure"] = Field(default="figure")
    caption: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: int = None

class ListBlock(BaseModel):
    type: Literal["list"] = Field(default="list")
    items: List[str]
    page_no: Optional[Union[int, List[int]]] = None
    token_count: int = None

class CellType(str, Enum):
    DATA = "data"
    HEADER = "header"
    COL_HEADER = "col_header"
    ROW_HEADER = "row_header"

class TableCell(BaseModel):
    text: str
    pos: List[int] = Field(..., description="[row, col] position")
    type: CellType
    span: Optional[List[int]] = Field(None, description="[row_span, col_span] if > 1")

class TableBlock(BaseModel):
    type: Literal["table"] = Field(default="table")
    caption: Optional[str] = None
    #cells: List[TableCell]
    cells: str
    dimensions: List[int] = Field(..., description="[max_rows, max_cols]")
    page_no: Optional[Union[int, List[int]]] = None
    token_count: int = None

ContentBlock = Union[Paragraph, CodeBlock, Figure, ListBlock, TableBlock]

class Children(BaseModel):
    title: Optional[str] = None
    level: int = None
    content: List[ContentBlock] = []
    children: List["Children"] = []

class Document(BaseModel):
    file_name: Optional[str] = None
    title: Optional[str] = None
    level: int = 0
    content: List[ContentBlock] = []
    children: List[Children] = []

# DocumentMD models
class Segment(BaseModel):
    content: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: Optional[int] = None
    segment_number: int
    segment_id: str  # "segment-md5 hash"
    provenance_path: str
    label: Literal["text", "table", "figure"] = "text"

class ChildrenMD(BaseModel):
    title: Optional[str] = None
    level: Optional[int] = None
    content: List[Segment]  # Always List[Segment] for consistency
    child_id: str  # "child-md5 hash"
    children: List["ChildrenMD"] = []
    token_count: Optional[int] = None
    page_no: Optional[Union[int, List[int]]] = None

class DocumentMD(BaseModel):
    file_name: Optional[str] = None
    title: Optional[str] = None
    level: int = 0
    content: List[Segment]  # Always List[Segment] for consistency
    children: List[ChildrenMD] = []
    token_count: Optional[int] = None
    doc_id: str  # "document-md5 hash"

# Rebuild models for forward references
Children.model_rebuild()
Document.model_rebuild()
ChildrenMD.model_rebuild()
DocumentMD.model_rebuild()

