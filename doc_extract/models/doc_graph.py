from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
from enum import Enum
import uuid

class NodeType(str, Enum):
    DOCUMENT = "Document"
    SECTION = "Section"
    PARAGRAPH = "Paragraph"
    TABLE = "Table"
    FIGURE = "Figure"
    CODE_BLOCK = "CodeBlock"
    LIST = "List"

class RelationshipType(str, Enum):
    CONTAINS = "CONTAINS"
    FOLLOWS = "FOLLOWS"

# Base classes
class GraphNode(BaseModel):
    """Base class for all graph nodes"""
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphRelationship(BaseModel):
    """Base class for all graph relationships"""
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    relationship_type: RelationshipType
    start_node_id: str
    end_node_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)

# Specific node models
class DocumentNode(GraphNode):
    """Document node in the graph"""
    node_type: Literal["Document"] = "Document"
    file_name: Optional[str] = None
    title: Optional[str] = None
    total_pages: int = 0
    language: str = "unknown"

class SectionNode(GraphNode):
    """Section node - handles all hierarchical levels"""
    node_type: Literal["Section"] = "Section"
    section_id: str
    title: Optional[str] = None
    level: Optional[int] = None
    order_index: int = 0
    token_count: int = 0  # UPDATED: Replaced word_count with token_count

class ParagraphNode(GraphNode):
    """Paragraph node in the graph"""
    node_type: Literal["Paragraph"] = "Paragraph"
    paragraph_id: str
    text: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: Optional[int] = None

class TableNode(GraphNode):
    """Table node in the graph"""
    node_type: Literal["Table"] = "Table"
    table_id: str
    caption: Optional[str] = None
    cells: str = "[]"  # JSON string of cells
    dimensions: List[int] = Field(default_factory=lambda: [0, 0])  # [max_rows, max_cols]
    page_no: Optional[Union[int, List[int]]] = None
    token_count: Optional[int] = None

class FigureNode(GraphNode):
    """Figure node in the graph"""
    node_type: Literal["Figure"] = "Figure"
    figure_id: str
    caption: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: Optional[int] = None

class CodeBlockNode(GraphNode):
    """Code block node in the graph"""
    node_type: Literal["CodeBlock"] = "CodeBlock"
    code_block_id: str
    text: str
    page_no: Optional[Union[int, List[int]]] = None
    token_count: Optional[int] = None

class ListNode(GraphNode):
    """List node in the graph"""
    node_type: Literal["List"] = "List"
    list_id: str
    items: List[str]
    page_no: Optional[Union[int, List[int]]] = None
    token_count: Optional[int] = None

# Relationship models
class ContainsRelationship(GraphRelationship):
    """CONTAINS relationship - handles ALL containment"""
    relationship_type: Literal["CONTAINS"] = "CONTAINS"
    order: int = 0

class FollowsRelationship(GraphRelationship):
    """FOLLOWS relationship between sequential nodes at the same level"""
    relationship_type: Literal["FOLLOWS"] = "FOLLOWS"
    sequence: int = 0

# Main graph model
class DocumentGraph(BaseModel):
    """Complete document graph structure"""
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_node: DocumentNode
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes.append(node)

    def add_relationship(self, relationship: GraphRelationship):
        """Add a relationship to the graph"""
        self.relationships.append(relationship)

    def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes including document node"""
        return [self.document_node] + self.nodes

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type"""
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.node_type == node_type]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        all_nodes = self.get_all_nodes()
        node_counts = {}
        for node in all_nodes:
            node_type = node.node_type
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        rel_counts = {}
        for rel in self.relationships:
            rel_type = rel.relationship_type
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

        return {
            "total_nodes": len(all_nodes),
            "total_relationships": len(self.relationships),
            "node_counts": node_counts,
            "relationship_counts": rel_counts
        }
