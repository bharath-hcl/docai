
from typing import Any
from doc_extract.base import PDF_Parser

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel import vlm_model_specs
from doc_extract.docling_.config import artifacts_path

class DoclingConverter(PDF_Parser):

    def __init__(self, source:str):
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
        return self.doc.document.export_to_dict()

    def ExportMarkDown(self) -> str:
        return self.doc.document.export_to_markdown()
    
def extract_to_dict(source):
    doc_conv = DoclingConverter(source=source)
    doc_dict = doc_conv.ExportDict()
    return doc_dict

def extract_to_markdown(source):
    doc_conv = DoclingConverter(source=source)
    doc_md = doc_conv.ExportMarkDown()
    return doc_md

