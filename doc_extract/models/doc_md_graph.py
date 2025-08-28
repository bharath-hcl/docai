import uuid
import json
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from collections import defaultdict

# Import your existing models
from doc_extract.models.document import DocumentMD, ChildrenMD, Segment

class NodeType(str, Enum):
    DOCUMENT = "Document"
    CHILD = "Child"
    SEGMENT = "Segment"

class RelationshipType(str, Enum):
    CONTAINS = "CONTAINS"
    FOLLOWS = "FOLLOWS"

# Base classes
class GraphNode(BaseModel):
    """Base class for all graph nodes"""
    node_id: str  # Will use the original meaningful ID (doc_id, child_id, segment_id)
    node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert to Neo4j node format"""
        return {
            "id": self.node_id,
            "labels": [self.node_type.value],
            "properties": self.properties
        }

class DocumentNode(GraphNode):
    """Document node in the graph"""
    node_type: NodeType = Field(default=NodeType.DOCUMENT)
    
    def __init__(self, document: DocumentMD, **kwargs):
        properties = {
            "file_name": document.file_name,
            "title": document.title,
            "level": document.level,
            "token_count": document.token_count
        }
        # Use doc_id as node_id directly
        super().__init__(node_id=document.doc_id, properties=properties, **kwargs)

class ChildNode(GraphNode):
    """Child node in the graph"""
    node_type: NodeType = Field(default=NodeType.CHILD)
    
    def __init__(self, child: ChildrenMD, **kwargs):
        properties = {
            "title": child.title,
            "level": child.level,
            "token_count": child.token_count,
            "page_no": child.page_no
        }
        # Use child_id as node_id directly
        super().__init__(node_id=child.child_id, properties=properties, **kwargs)

class SegmentNode(GraphNode):
    """Segment node in the graph"""
    node_type: NodeType = Field(default=NodeType.SEGMENT)
    
    def __init__(self, segment: Segment, **kwargs):
        properties = {
            "content": segment.content,
            "segment_number": segment.segment_number,
            "provenance_path": segment.provenance_path,
            "label": segment.label,
            "token_count": segment.token_count,
            "page_no": segment.page_no
        }
        # Use segment_id as node_id directly
        super().__init__(node_id=segment.segment_id, properties=properties, **kwargs)

class GraphRelationship(BaseModel):
    """Represents a relationship between two nodes"""
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    def to_neo4j_relationship(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship format"""
        return {
            "id": self.relationship_id,
            "start_node": self.source_node_id,
            "end_node": self.target_node_id,
            "type": self.relationship_type.value,
            "properties": self.properties
        }

class DocumentGraph(BaseModel):
    """Main graph class that converts DocumentMD to graph structure"""
    nodes: Dict[str, GraphNode] = Field(default_factory=dict)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    document_id: Optional[str] = None
    
    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        return node.node_id
    
    def add_relationship(self, source_id: str, target_id: str, 
                        relationship_type: RelationshipType, 
                        properties: Dict[str, Any] = None) -> str:
        """Add a relationship between two nodes"""
        if properties is None:
            properties = {}
        
        relationship = GraphRelationship(
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_type=relationship_type,
            properties=properties
        )
        self.relationships.append(relationship)
        return relationship.relationship_id
    
    def from_document_md(self, document: DocumentMD) -> 'DocumentGraph':
        """Convert DocumentMD to graph structure"""
        self.document_id = document.doc_id
        
        # Create document node using doc_id as node_id
        doc_node = DocumentNode(document)
        doc_node_id = self.add_node(doc_node)
        
        # Process document content (segments)
        prev_segment_id = None
        for segment in document.content:
            segment_node = SegmentNode(segment)
            segment_node_id = self.add_node(segment_node)
            
            # Document CONTAINS segment
            self.add_relationship(doc_node_id, segment_node_id, RelationshipType.CONTAINS)
            
            # Segment FOLLOWS previous segment
            if prev_segment_id:
                self.add_relationship(prev_segment_id, segment_node_id, RelationshipType.FOLLOWS)
            
            prev_segment_id = segment_node_id
        
        # Process children recursively
        prev_child_id = None
        for child in document.children:
            child_node_id = self._process_child(child, doc_node_id)
            
            # Child FOLLOWS previous child
            if prev_child_id:
                self.add_relationship(prev_child_id, child_node_id, RelationshipType.FOLLOWS)
            
            prev_child_id = child_node_id
        
        return self
    
    def _process_child(self, child: ChildrenMD, parent_node_id: str) -> str:
        """Recursively process child nodes"""
        # Create child node using child_id as node_id
        child_node = ChildNode(child)
        child_node_id = self.add_node(child_node)
        
        # Parent CONTAINS child
        self.add_relationship(parent_node_id, child_node_id, RelationshipType.CONTAINS)
        
        # Process child content (segments)
        prev_segment_id = None
        for segment in child.content:
            segment_node = SegmentNode(segment)
            segment_node_id = self.add_node(segment_node)
            
            # Child CONTAINS segment
            self.add_relationship(child_node_id, segment_node_id, RelationshipType.CONTAINS)
            
            # Segment FOLLOWS previous segment
            if prev_segment_id:
                self.add_relationship(prev_segment_id, segment_node_id, RelationshipType.FOLLOWS)
            
            prev_segment_id = segment_node_id
        
        # Process nested children
        prev_nested_child_id = None
        for nested_child in child.children:
            nested_child_node_id = self._process_child(nested_child, child_node_id)
            
            # Nested child FOLLOWS previous nested child
            if prev_nested_child_id:
                self.add_relationship(prev_nested_child_id, nested_child_node_id, RelationshipType.FOLLOWS)
            
            prev_nested_child_id = nested_child_node_id
        
        return child_node_id
        return child_node_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary format"""
        return {
            "document_id": self.document_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "properties": node.properties
                }
                for node in self.nodes.values()
            ],
            "relationships": [
                {
                    "relationship_id": rel.relationship_id,
                    "source_node_id": rel.source_node_id,
                    "target_node_id": rel.target_node_id,
                    "relationship_type": rel.relationship_type.value,
                    "properties": rel.properties
                }
                for rel in self.relationships
            ]
        }
    
    def to_json(self, file_path: Optional[str] = None, indent: int = 2) -> str:
        """Convert graph to JSON format"""
        json_data = json.dumps(self.to_dict(), indent=indent)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
        
        return json_data
    
    def to_neo4j_format(self) -> Dict[str, Any]:
        """Convert graph to Neo4j import format"""
        return {
            "nodes": [node.to_neo4j_node() for node in self.nodes.values()],
            "relationships": [rel.to_neo4j_relationship() for rel in self.relationships]
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        node_counts = defaultdict(int)
        relationship_counts = defaultdict(int)
        
        for node in self.nodes.values():
            node_counts[node.node_type.value] += 1
        
        for rel in self.relationships:
            relationship_counts[rel.relationship_type.value] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "node_counts": dict(node_counts),
            "relationship_counts": dict(relationship_counts)
        }
    
    def visualize_structure(self) -> str:
        """Create a simple text visualization of the graph structure"""
        lines = []
        lines.append(f"Document Graph - ID: {self.document_id}")
        lines.append("=" * 50)
        
        stats = self.get_graph_statistics()
        lines.append(f"Nodes: {stats['total_nodes']}")
        lines.append(f"Relationships: {stats['total_relationships']}")
        lines.append("")
        
        for node_type, count in stats['node_counts'].items():
            lines.append(f"{node_type}: {count}")
        
        lines.append("")
        for rel_type, count in stats['relationship_counts'].items():
            lines.append(f"{rel_type}: {count}")
        
        return "\n".join(lines)

# Usage example and utility functions
class DocumentGraphBuilder:
    """Utility class to build graphs from DocumentMD"""
    
    @staticmethod
    def build_graph(document: DocumentMD) -> DocumentGraph:
        """Build a graph from DocumentMD"""
        graph = DocumentGraph()
        return graph.from_document_md(document)
    
    @staticmethod
    def save_graph_json(graph: DocumentGraph, file_path: str):
        """Save graph to JSON file"""
        graph.to_json(file_path)
    
    @staticmethod
    def load_graph_from_json(file_path: str) -> Dict[str, Any]:
        """Load graph from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

