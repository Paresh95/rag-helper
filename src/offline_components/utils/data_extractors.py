from typing import Dict, List, Union
from pathlib import Path
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


class DoclingPDFConverter:
    def __init__(self):
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )

    def extract_pdf_file(self, file_path: Union[str, Path]) -> Dict:
        """Extract a single PDF to a JSON dict."""
        result = self.converter.convert(source=str(file_path))
        return result.document.export_to_dict()

    def extract_pdf_batch(self, file_paths: List[Union[str, Path]]) -> List[Dict]:
        """Extract multiple PDFs in one call."""
        results = self.converter.convert(source=[str(p) for p in file_paths])
        return [doc.document.export_to_dict() for doc in results]
