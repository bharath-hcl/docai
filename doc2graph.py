import json
import uuid
from typing import Dict, List, Any
from neo4j import GraphDatabase
from dataclasses import dataclass

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

class DocumentGraphCreator:
    def __init__(self, neo4j_config: Neo4jConfig):
        self.driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password)
        )
        self.neo4j_config = neo4j_config
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def create_indexes(self):
        """Create indexes for better performance."""
        indexes = [
            "CREATE INDEX document_id_idx IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
            "CREATE INDEX document_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX section_id_idx IF NOT EXISTS FOR (s:Section) ON (s.section_id)",
            "CREATE INDEX section_title_idx IF NOT EXISTS FOR (s:Section) ON (s.title)",
            "CREATE INDEX subsection_id_idx IF NOT EXISTS FOR (ss:SubSection) ON (ss.sub_section_id)",
            "CREATE INDEX paragraph_id_idx IF NOT EXISTS FOR (p:Paragraph) ON (p.paragraph_id)",
            "CREATE INDEX author_name_idx IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX table_id_idx IF NOT EXISTS FOR (t:Table) ON (t.table_id)",
            "CREATE INDEX figure_id_idx IF NOT EXISTS FOR (f:Figure) ON (f.figure_id)"
        ]
        
        with self.driver.session(database=self.neo4j_config.database) as session:
            for index_query in indexes:
                session.run(index_query)
        print("Indexes created.")

    def calculate_total_pages(self, json_data: Dict[str, Any]) -> int:
        """Calculate total pages by finding max page_no in the document."""
        max_page = 0
        
        def find_max_page(sections):
            nonlocal max_page
            for section in sections:
                # Check content items
                for content_item in section.get('content', []):
                    page_no = content_item.get('page_no')
                    if page_no and isinstance(page_no, int):
                        max_page = max(max_page, page_no)
                
                # Check sub-sections recursively
                if 'sub_sections' in section:
                    find_max_page(section['sub_sections'])
        
        if 'sections' in json_data:
            find_max_page(json_data['sections'])
        
        return max_page

    def create_document_node(self, session, json_data: Dict[str, Any]) -> str:
        """Create the main document node."""
        document_id = json_data.get('document_id', f'doc_{uuid.uuid4().hex[:8]}')
        title = json_data.get('title', 'Untitled Document')
        total_pages = self.calculate_total_pages(json_data)
        
        cypher = """
        CREATE (d:Document {
            document_id: $document_id,
            title: $title,
            document_type: $document_type,
            total_pages: $total_pages,
            language: $language
        })
        RETURN d.document_id as id
        """
        
        result = session.run(cypher, 
            document_id=document_id,
            title=title,
            document_type='document',  # Generic type
            total_pages=total_pages,
            language='unknown'  # Don't assume language
        )
        
        return result.single()['id']

    def create_section_node(self, session, section: Dict[str, Any], order_index: int) -> str:
        """Create a section node."""
        section_id = f"section_{uuid.uuid4().hex[:8]}"
        
        cypher = """
        CREATE (s:Section {
            section_id: $section_id,
            title: $title,
            level: $level,
            order_index: $order_index
        })
        RETURN s.section_id as id
        """
        
        result = session.run(cypher,
            section_id=section_id,
            title=section.get('title', ''),
            level=section.get('level', 1),
            order_index=order_index
        )
        
        return result.single()['id']

    def create_subsection_node(self, session, subsection: Dict[str, Any], order_index: int) -> str:
        """Create a subsection node."""
        subsection_id = f"subsection_{uuid.uuid4().hex[:8]}"
        
        # Calculate word count from content
        word_count = 0
        for content_item in subsection.get('content', []):
            if content_item.get('type') == 'paragraph':
                text = content_item.get('text', '')
                word_count += len(text.split())
        
        cypher = """
        CREATE (ss:SubSection {
            sub_section_id: $sub_section_id,
            title: $title,
            level: $level,
            word_count: $word_count
        })
        RETURN ss.sub_section_id as id
        """
        
        result = session.run(cypher,
            sub_section_id=subsection_id,
            title=subsection.get('title', ''),
            level=subsection.get('level', 2),
            word_count=word_count
        )
        
        return result.single()['id']

    def create_content_nodes(self, session, content_items: List[Dict[str, Any]], parent_id: str):
        """Create content nodes (paragraphs, tables, figures) and link to parent."""
        content_ids = []
        
        for order_index, item in enumerate(content_items):
            content_type = item.get('type', '')
            content_id = None
            
            if content_type == 'paragraph':
                content_id = self.create_paragraph_node(session, item)
                
            elif content_type == 'table':
                content_id = self.create_table_node(session, item)
                
            elif content_type == 'figure':
                content_id = self.create_figure_node(session, item)
            
            # Create CONTAINS relationship if content was created
            if content_id:
                content_ids.append(content_id)
                self.create_contains_relationship(session, parent_id, content_id, order_index)
        
        # Create FOLLOWS relationships between content items
        for i in range(len(content_ids) - 1):
            self.create_follows_relationship(session, content_ids[i], content_ids[i + 1], i)

    def create_paragraph_node(self, session, paragraph: Dict[str, Any]) -> str:
        """Create a paragraph node."""
        paragraph_id = f"paragraph_{uuid.uuid4().hex[:8]}"
        
        cypher = """
        CREATE (p:Paragraph {
            paragraph_id: $paragraph_id,
            text: $text,
            page_no: $page_no
        })
        RETURN p.paragraph_id as id
        """
        
        result = session.run(cypher,
            paragraph_id=paragraph_id,
            text=paragraph.get('text', ''),
            page_no=paragraph.get('page_no')
        )
        
        return result.single()['id']

    def create_table_node(self, session, table: Dict[str, Any]) -> str:
        """Create a table node."""
        table_id = f"table_{uuid.uuid4().hex[:8]}"
        
        cells = table.get('cells', [])
        cell_count = len(cells)
        
        # Calculate rows and columns from cells if available
        max_row = 0
        max_col = 0
        if cells:
            for cell in cells:
                if isinstance(cell, dict):
                    end_row = cell.get('end_row_offset_idx', 0)
                    end_col = cell.get('end_col_offset_idx', 0)
                    if isinstance(end_row, int):
                        max_row = max(max_row, end_row)
                    if isinstance(end_col, int):
                        max_col = max(max_col, end_col)
        
        cypher = """
        CREATE (t:Table {
            table_id: $table_id,
            caption: $caption,
            cell_count: $cell_count,
            rows: $rows,
            columns: $columns,
            cells: $cells
        })
        RETURN t.table_id as id
        """
        
        result = session.run(cypher,
            table_id=table_id,
            caption=table.get('caption', ''),
            cell_count=cell_count,
            rows=max_row + 1 if max_row > 0 else 0,
            columns=max_col + 1 if max_col > 0 else 0,
            cells=json.dumps(cells) if cells else '[]'
        )
        
        return result.single()['id']

    def create_figure_node(self, session, figure: Dict[str, Any]) -> str:
        """Create a figure node."""
        figure_id = f"figure_{uuid.uuid4().hex[:8]}"
        
        cypher = """
        CREATE (f:Figure {
            figure_id: $figure_id,
            caption: $caption
        })
        RETURN f.figure_id as id
        """
        
        result = session.run(cypher,
            figure_id=figure_id,
            caption=figure.get('caption', '')
        )
        
        return result.single()['id']

    def create_contains_relationship(self, session, parent_id: str, child_id: str, order: int):
        """Create a CONTAINS relationship."""
        cypher = """
        MATCH (parent), (child)
        WHERE (parent.document_id = $parent_id OR parent.section_id = $parent_id OR parent.sub_section_id = $parent_id)
        AND (child.section_id = $child_id OR child.sub_section_id = $child_id OR 
             child.paragraph_id = $child_id OR child.table_id = $child_id OR child.figure_id = $child_id)
        CREATE (parent)-[:CONTAINS {order: $order}]->(child)
        """
        
        session.run(cypher, parent_id=parent_id, child_id=child_id, order=order)

    def create_follows_relationship(self, session, predecessor_id: str, successor_id: str, sequence: int):
        """Create a FOLLOWS relationship where successor follows predecessor."""
        cypher = """
        MATCH (predecessor), (successor)
        WHERE (predecessor.section_id = $predecessor_id OR predecessor.sub_section_id = $predecessor_id OR 
            predecessor.paragraph_id = $predecessor_id OR predecessor.table_id = $predecessor_id OR predecessor.figure_id = $predecessor_id)
        AND (successor.section_id = $successor_id OR successor.sub_section_id = $successor_id OR 
            successor.paragraph_id = $successor_id OR successor.table_id = $successor_id OR successor.figure_id = $successor_id)
        CREATE (successor)-[:FOLLOWS {sequence: $sequence}]->(predecessor)
        """
        
        session.run(cypher, predecessor_id=predecessor_id, successor_id=successor_id, sequence=sequence)


    def process_sections(self, session, sections: List[Dict[str, Any]], parent_id: str, is_subsection: bool = False):
        """Process sections or subsections recursively."""
        section_ids = []
        
        for section_index, section in enumerate(sections):
            # Create section or subsection node
            if is_subsection:
                section_id = self.create_subsection_node(session, section, section_index)
            else:
                section_id = self.create_section_node(session, section, section_index)
            
            section_ids.append(section_id)
            
            # Create CONTAINS relationship from parent to section
            self.create_contains_relationship(session, parent_id, section_id, section_index)
            
            # Process section content
            if 'content' in section:
                self.create_content_nodes(session, section['content'], section_id)
            
            # Process sub-sections
            if 'sub_sections' in section and section['sub_sections']:
                self.process_sections(session, section['sub_sections'], section_id, is_subsection=True)
        
        # Create FOLLOWS relationships between sections at the same level
        for i in range(len(section_ids) - 1):
            self.create_follows_relationship(session, section_ids[i], section_ids[i + 1], i)

    def process_json_to_graph(self, json_file_path: str):
        """Main method to process JSON and create the graph."""
        print("Loading JSON file...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        with self.driver.session(database=self.neo4j_config.database) as session:
            print("Creating document node...")
            document_id = self.create_document_node(session, json_data)
            
            print("Processing sections...")
            if 'sections' in json_data and json_data['sections']:
                self.process_sections(session, json_data['sections'], document_id)
        
        print("Graph creation completed successfully!")

def main():
    # Configuration
    neo4j_config = Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="hcltech_graph_2025"  # Update with your password
    )
    
    json_file_path = "docling_rag.json"  # Update with your JSON file path
    
    with DocumentGraphCreator(neo4j_config) as creator:
        # Optional: Clear existing data
        creator.clear_database()
        
        # Create indexes
        creator.create_indexes()
        
        # Process JSON and create graph
        creator.process_json_to_graph(json_file_path)
        
        print("\nGraph Statistics:")
        with creator.driver.session() as session:
            # Count nodes by type
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC")
            for record in result:
                print(f"- {record['type']}: {record['count']}")
            
            print("\nRelationship Statistics:")
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
            for record in result:
                print(f"- {record['rel_type']}: {record['count']}")

if __name__ == "__main__":
    main()
