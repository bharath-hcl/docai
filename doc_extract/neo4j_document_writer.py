# neo4j_document_writer.py (Fixed Version)

from typing import List, Optional, Dict, Any
from neo4j import GraphDatabase
from dataclasses import dataclass
import json

# Import the graph classes
from doc_extract.models.doc_md_graph import (
    DocumentGraph, DocumentNode, ChildNode, SegmentNode,
    GraphRelationship, NodeType, RelationshipType, DocumentGraphBuilder
)

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

class Neo4jDocumentGraphWriter:
    """Handles writing DocumentGraph to Neo4j database with batch operations"""
    
    def __init__(self, neo4j_config: Neo4jConfig):
        self.driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password)
        )
        self.neo4j_config = neo4j_config
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the driver"""
        if self.driver:
            self.driver.close()
    
    def close(self):
        """Manually close the driver"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    def create_indexes(self):
        """Create indexes for better performance"""
        indexes = [
            # Single node_id index for each type (simplified approach)
            "CREATE INDEX document_node_id_idx IF NOT EXISTS FOR (d:Document) ON (d.node_id)",
            "CREATE INDEX child_node_id_idx IF NOT EXISTS FOR (c:Child) ON (c.node_id)", 
            "CREATE INDEX segment_node_id_idx IF NOT EXISTS FOR (s:Segment) ON (s.node_id)",
            
            # Additional useful indexes
            "CREATE INDEX document_file_name_idx IF NOT EXISTS FOR (d:Document) ON (d.file_name)",
            "CREATE INDEX segment_number_idx IF NOT EXISTS FOR (s:Segment) ON (s.segment_number)"
        ]
        
        with self.driver.session(database=self.neo4j_config.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    print(f"✅ Created index: {index_query.split('FOR')[0].split()[-1]}")
                except Exception as e:
                    print(f"⚠️ Index creation failed: {e}")
        print("Index creation completed.")
    
    def create_indexes_legacy(self):
        """Alternative: Create indexes without IF NOT EXISTS (for older Neo4j versions)"""
        indexes = [
            "CREATE INDEX FOR (d:Document) ON (d.node_id)",
            "CREATE INDEX FOR (c:Child) ON (c.node_id)",
            "CREATE INDEX FOR (s:Segment) ON (s.node_id)",
            "CREATE INDEX FOR (d:Document) ON (d.file_name)",
            "CREATE INDEX FOR (s:Segment) ON (s.segment_number)"
        ]
        
        with self.driver.session(database=self.neo4j_config.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    print(f"✅ Created index")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"⚠️ Index creation failed: {e}")
        print("Index creation completed.")
    
    def write_graph_to_neo4j(self, graph: DocumentGraph):
        """Write the complete DocumentGraph to Neo4j database using batch operations"""
        with self.driver.session(database=self.neo4j_config.database) as session:
            print("Creating nodes in batches...")
            self._batch_create_nodes(session, list(graph.nodes.values()))
            
            print("Creating relationships in batches...")
            self._batch_create_relationships(session, graph.relationships)
            
            print("Graph successfully written to Neo4j!")
    
    def _batch_create_nodes(self, session, nodes: List):
        """Create all nodes using batch operations with UNWIND"""
        # Group nodes by type for efficient batch processing
        nodes_by_type = {}
        for node in nodes:
            node_type = node.node_type.value
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        # Create each node type in batches
        for node_type, node_list in nodes_by_type.items():
            self._create_node_batch(session, node_type, node_list)
    
    def _create_node_batch(self, session, node_type: str, nodes: List):
        """Create a batch of nodes of the same type"""
        chunk_size = 100
        
        for i in range(0, len(nodes), chunk_size):
            chunk = nodes[i:i + chunk_size]
            props_list = []
            
            for node in chunk:
                # Use the meaningful ID directly as node_id
                node_props = {
                    "node_id": node.node_id,  # This is doc_id/child_id/segment_id
                    **node.properties
                }
                
                # Clean up None values
                node_props = {k: v for k, v in node_props.items() if v is not None}
                
                # Convert lists to JSON strings for Neo4j compatibility
                for key, value in node_props.items():
                    if isinstance(value, list):
                        node_props[key] = json.dumps(value)
                
                props_list.append(node_props)
            
            # Use UNWIND with MERGE for batch creation
            cypher = f"""
            UNWIND $props AS prop
            MERGE (n:{node_type} {{node_id: prop.node_id}})
            SET n += prop
            """
            
            try:
                session.run(cypher, props=props_list)
                print(f"Created {len(chunk)} {node_type} nodes")
            except Exception as e:
                print(f"❌ Error creating {node_type} nodes: {e}")
                if props_list:
                    print(f"Sample props: {props_list[0]}")
    
    def _batch_create_relationships(self, session, relationships: List[GraphRelationship]):
        """Create all relationships using batch operations"""
        # Group relationships by type
        rels_by_type = {}
        for rel in relationships:
            rel_type = rel.relationship_type.value
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)
        
        print(f"📊 Relationship breakdown: {[(k, len(v)) for k, v in rels_by_type.items()]}")
        
        # Create each relationship type in batches
        for rel_type, rel_list in rels_by_type.items():
            self._create_relationship_batch(session, rel_type, rel_list)
    
    def _create_relationship_batch(self, session, rel_type: str, relationships: List[GraphRelationship]):
        """Create a batch of relationships of the same type"""
        chunk_size = 100
        total_created = 0
        
        for i in range(0, len(relationships), chunk_size):
            chunk = relationships[i:i + chunk_size]
            rel_props_list = []
            
            for rel in chunk:
                rel_props = {
                    "start_id": rel.source_node_id,
                    "end_id": rel.target_node_id
                }
                
                # Add relationship-specific properties
                if rel.properties:
                    rel_props.update(rel.properties)
                
                rel_props_list.append(rel_props)
            
            # Create cypher query based on relationship type
            if rel_type == "CONTAINS":
                cypher = """
                UNWIND $rels AS rel
                MATCH (start {node_id: rel.start_id}), (end {node_id: rel.end_id})
                CREATE (start)-[:CONTAINS]->(end)
                RETURN count(*) as created
                """
            elif rel_type == "FOLLOWS":
                cypher = """
                UNWIND $rels AS rel
                MATCH (start {node_id: rel.start_id}), (end {node_id: rel.end_id})
                CREATE (start)-[:FOLLOWS]->(end)
                RETURN count(*) as created
                """
            else:
                # Generic relationship creation
                cypher = f"""
                UNWIND $rels AS rel
                MATCH (start {{node_id: rel.start_id}}), (end {{node_id: rel.end_id}})
                CREATE (start)-[:{rel_type}]->(end)
                RETURN count(*) as created
                """
            
            # Execute with proper error handling
            try:
                result = session.run(cypher, rels=rel_props_list)
                batch_created = result.single()["created"]
                total_created += batch_created
                print(f"✅ Created {batch_created} {rel_type} relationships (batch {i//chunk_size + 1})")
            except Exception as e:
                print(f"❌ Error creating {rel_type} relationships: {e}")
                if rel_props_list:
                    print(f"Sample params: {rel_props_list[0]}")
        
        print(f"🎯 Total {rel_type} relationships created: {total_created}")
    
    def get_graph_statistics(self) -> dict:
        """Get statistics from the Neo4j database"""
        with self.driver.session(database=self.neo4j_config.database) as session:
            # Count nodes by type
            node_stats = {}
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC")
            for record in result:
                node_stats[record['type']] = record['count']
            
            # Count relationships
            rel_stats = {}
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
            for record in result:
                rel_stats[record['rel_type']] = record['count']
            
            return {
                "nodes": node_stats,
                "relationships": rel_stats,
                "total_nodes": sum(node_stats.values()),
                "total_relationships": sum(rel_stats.values())
            }

# Example usage and utility functions
class DocumentGraphNeo4jManager:
    """High-level manager for DocumentMD to Neo4j operations"""
    
    def __init__(self, neo4j_config: Neo4jConfig):
        self.neo4j_config = neo4j_config
    
    def setup_database(self, clear_existing: bool = False):
        """Setup the database with indexes"""
        with Neo4jDocumentGraphWriter(self.neo4j_config) as writer:
            if clear_existing:
                writer.clear_database()
            writer.create_indexes()
    
    def import_document(self, document_md, clear_existing: bool = False):
        """Import a DocumentMD into Neo4j"""
        
        
        # Build graph from DocumentMD
        graph = DocumentGraphBuilder.build_graph(document_md)
        
        # Write to Neo4j
        with Neo4jDocumentGraphWriter(self.neo4j_config) as writer:
            if clear_existing:
                writer.clear_database()
                writer.create_indexes()
            
            writer.write_graph_to_neo4j(graph)
            
            # Get and print statistics
            stats = writer.get_graph_statistics()
            print("📊 Import Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return graph

# Example usage
if __name__ == "__main__":
    # Configure Neo4j connection
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password",
        database="neo4j"
    )
    
    # Initialize manager
    manager = DocumentGraphNeo4jManager(config)
    
    # Setup database (create indexes)
    manager.setup_database(clear_existing=True)
    
    # Get statistics
    with Neo4jDocumentGraphWriter(config) as writer:
        stats = writer.get_graph_statistics()
        print("Database Statistics:", stats)
