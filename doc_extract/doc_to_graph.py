import uuid
import json
from typing import List

from doc_extract.models.document import Document, Children, ContentBlock, Paragraph, CodeBlock, Figure, ListBlock, TableBlock
from doc_extract.models.doc_graph import (DocumentGraph, DocumentNode, SectionNode,
                                         ContainsRelationship, FollowsRelationship, ParagraphNode,
                                         CodeBlockNode, FigureNode, ListNode, TableNode)

class DocumentToGraphConverter:
    """Convert Document Pydantic model to DocumentGraph"""

    def convert_document_to_graph(self, document: Document) -> DocumentGraph:
        """Convert Document model to DocumentGraph model"""
        document_node = DocumentNode(
            file_name=document.file_name,
            title=document.title or "Untitled Document",
            total_pages=self._calculate_total_pages(document)
        )

        graph = DocumentGraph(document_node=document_node)

        # Process root-level content first
        if document.content:
            self._process_content(document.content, document_node.node_id, graph)

        # Process hierarchical children
        if document.children:
            self._process_children_recursive(
                document.children,
                document_node.node_id,
                graph
            )

        return graph

    def _calculate_total_pages(self, document: Document) -> int:
        """Calculate total pages from document content"""
        max_page = 0

        def extract_max_page_from_content(content: List[ContentBlock]):
            nonlocal max_page
            for item in content:
                page_no = item.page_no
                if page_no:
                    if isinstance(page_no, int):
                        max_page = max(max_page, page_no)
                    elif isinstance(page_no, list):
                        max_page = max(max_page, max(page_no))

        def extract_max_page_from_children(children: List[Children]):
            nonlocal max_page
            for child in children:
                # Check content pages
                extract_max_page_from_content(child.content)
                # Check nested children
                if child.children:
                    extract_max_page_from_children(child.children)

        # Check root-level content
        extract_max_page_from_content(document.content)
        
        # Check children content
        if document.children:
            extract_max_page_from_children(document.children)

        return max_page

    def _calculate_section_token_count(self, child: Children) -> int:
        """Calculate token count for a section by summing all child content and nested section token counts"""
        total_tokens = 0
        
        # Sum token counts from direct content
        for content_item in child.content:
            if content_item.token_count:
                total_tokens += content_item.token_count
        
        # Recursively sum token counts from nested children
        if child.children:
            for nested_child in child.children:
                total_tokens += self._calculate_section_token_count(nested_child)
        
        return total_tokens

    def _process_children_recursive(self, children: List[Children], parent_id: str, graph: DocumentGraph):
        """Process children (sections) recursively with token_count calculation"""
        section_nodes = []
        for order_index, child in enumerate(children):
            # UPDATED: Calculate token_count by summing all child content and nested section tokens
            section_token_count = self._calculate_section_token_count(child)

            section_node = SectionNode(
                section_id=f"section_{uuid.uuid4().hex[:8]}",
                title=child.title,
                level=child.level,
                order_index=order_index,
                token_count=section_token_count  # UPDATED: Use token_count instead of word_count
            )

            graph.add_node(section_node)
            section_nodes.append(section_node)

            # Create CONTAINS relationship
            contains_rel = ContainsRelationship(
                start_node_id=parent_id,
                end_node_id=section_node.node_id,
                order=order_index
            )
            graph.add_relationship(contains_rel)

            # Process section content
            if child.content:
                self._process_content(child.content, section_node.node_id, graph)

            # Process nested children recursively
            if child.children:
                self._process_children_recursive(
                    child.children,
                    section_node.node_id,
                    graph
                )

        # Create FOLLOWS relationships between sections
        for i in range(len(section_nodes) - 1):
            follows_rel = FollowsRelationship(
                start_node_id=section_nodes[i + 1].node_id,
                end_node_id=section_nodes[i].node_id,
                sequence=i
            )
            graph.add_relationship(follows_rel)

    def _process_content(self, content: List[ContentBlock], parent_id: str, graph: DocumentGraph):
        """Process content items and create appropriate nodes"""
        content_nodes = []

        for order_index, item in enumerate(content):
            content_node = None

            if isinstance(item, Paragraph):
                content_node = ParagraphNode(
                    paragraph_id=f"paragraph_{uuid.uuid4().hex[:8]}",
                    text=item.text,
                    page_no=item.page_no,
                    token_count=item.token_count
                )

            elif isinstance(item, CodeBlock):
                content_node = CodeBlockNode(
                    code_block_id=f"code_{uuid.uuid4().hex[:8]}",
                    text=item.text,
                    page_no=item.page_no,
                    token_count=item.token_count
                )

            elif isinstance(item, Figure):
                content_node = FigureNode(
                    figure_id=f"figure_{uuid.uuid4().hex[:8]}",
                    caption=item.caption,
                    page_no=item.page_no,
                    token_count=item.token_count
                )

            elif isinstance(item, ListBlock):
                content_node = ListNode(
                    list_id=f"list_{uuid.uuid4().hex[:8]}",
                    items=item.items,
                    page_no=item.page_no,
                    token_count=item.token_count
                )

            elif isinstance(item, TableBlock):
                # Use dimensions directly from the model
                max_rows, max_cols = item.dimensions if item.dimensions else [0, 0]
                
                content_node = TableNode(
                    table_id=f"table_{uuid.uuid4().hex[:8]}",
                    caption=item.caption,
                    cells=item.cells,  # Already a JSON string
                    dimensions=item.dimensions,
                    page_no=item.page_no,
                    token_count=item.token_count
                )

            if content_node:
                graph.add_node(content_node)
                content_nodes.append(content_node)

                # Create CONTAINS relationship
                contains_rel = ContainsRelationship(
                    start_node_id=parent_id,
                    end_node_id=content_node.node_id,
                    order=order_index
                )
                graph.add_relationship(contains_rel)

        # Create FOLLOWS relationships between content items
        for i in range(len(content_nodes) - 1):
            follows_rel = FollowsRelationship(
                start_node_id=content_nodes[i + 1].node_id,
                end_node_id=content_nodes[i].node_id,
                sequence=i
            )
            graph.add_relationship(follows_rel)
