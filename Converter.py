
from typing import Any
from doc_extract.base import PDF_Parser
import copy
import numpy as np
from sklearn.cluster import AgglomerativeClustering 
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel import vlm_model_specs
from doc_extract.Docling.config import artifacts_path

class DoclingConverter(PDF_Parser):

    def __init__(self, source:str, artifacts_path: str = None):
        if artifacts_path is None:
            self.artifacts_path = artifacts_path
        self.source = source
        self.use_smoldoc = False
        if self.use_smoldoc == True:
            # Use predefined SmolDocling configuration
            Vlmpipeline_options = VlmPipelineOptions(
                artifacts_path=artifacts_path,
                vlm_options=vlm_model_specs.SMOLDOCLING_TRANSFORMERS
            )
            self.doc_converter = DocumentConverter(
                                format_options={
                                    InputFormat.PDF: PdfFormatOption(
                                        pipeline_cls=VlmPipeline,
                                        pipeline_options=Vlmpipeline_options
                                    )
                                }
                            )
        else:
            pdf_pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
            self.doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
                }
            )
        self.doc = self._process(source)
    
    def _process(self, source:str):
        if self.use_smoldoc == False:
            result = self.doc_converter.convert(source)
        else:
            result = self.smoldoc_doc_converter.convert(source)
        return result

    def ExportDict(self) -> dict:
        doc_dict = self.doc.document.export_to_dict()
        doc_dict = Docling_adjust_section_header_levels(doc_dict)
        return doc_dict

    def ExportMarkDown(self) -> str:

        return self.doc.document.export_to_markdown()
    
def Docling_adjust_section_header_levels(doc_dict):
    """
    Adjusts the level values of section headers based on their visual height.
    
    Args:
        doc_dict (dict): Docling dictionary containing 'texts' key with text fragments
        
    Returns:
        dict: Modified dictionary with updated level values for section headers
    """
    # Create a deep copy to avoid modifying the original
    result = copy.deepcopy(doc_dict)
    
    # Extract heights and indices of section_header elements
    heights = []
    indices = []
    
    for idx, element in enumerate(result.get('texts', [])):
        if element.get('label') == 'section_header':
            bbox = element['prov'][0]['bbox']
            height = bbox['t'] - bbox['b']
            heights.append(height)
            indices.append(idx)
    
    # If no section headers found, return unchanged
    if not heights:
        return result
    
    # Convert to numpy array for clustering
    heights_array = np.array(heights).reshape(-1, 1)
    
    # Determine optimal number of clusters
    # Use a distance threshold approach for automatic cluster detection
    # The threshold is set relative to the height variation
    height_range = max(heights) - min(heights)
    distance_threshold = max(1.0, height_range * 0.1)  # 10% of height range, minimum 1.0
    
    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        linkage='ward'
    )
    
    try:
        cluster_labels = clustering.fit_predict(heights_array)
    except:
        # Fallback: if clustering fails, use simple thresholding
        unique_heights = sorted(set(heights), reverse=True)
        height_to_level = {height: idx + 1 for idx, height in enumerate(unique_heights)}
        
        for idx, height in zip(indices, heights):
            result['texts'][idx]['level'] = height_to_level[height]
        
        return result
    
    # Calculate mean height for each cluster
    cluster_means = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_means:
            cluster_means[label] = []
        cluster_means[label].append(heights[i])
    
    # Convert to mean heights
    for label in cluster_means:
        cluster_means[label] = np.mean(cluster_means[label])
    
    # Rank clusters by mean height (descending order)
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
    cluster_to_level = {cluster_id: level + 1 for level, (cluster_id, _) in enumerate(sorted_clusters)}
    
    # Update level values in the result dictionary
    for idx, cluster_label in zip(indices, cluster_labels):
        result['texts'][idx]['level'] = cluster_to_level[cluster_label]
    
    return result

def extract_to_dict(source, artifacts_path=artifacts_path):
    doc_conv = DoclingConverter(source=source, artifacts_path=artifacts_path)
    doc_dict = doc_conv.ExportDict()
    return doc_dict

def extract_to_markdown(source, artifacts_path=artifacts_path):
    doc_conv = DoclingConverter(source=source, artifacts_path=artifacts_path)
    doc_md = doc_conv.ExportMarkDown()
    return doc_md

