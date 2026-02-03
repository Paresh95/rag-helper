from typing import List
import yaml
from pathlib import Path
from src.utils.base import BaseIngester
from src.offline_components.utils.data_loaders import (
    get_files_to_handle,
    save_json_to_path,
    load_json_from_path,
)
from src.offline_components.utils.data_extractors import docling_pdf_to_json
from src.offline_components.utils.data_processors import parse_docling_json
from tqdm import tqdm


class Ingester(BaseIngester):
    """
    Extract raw data, process data, create chunks, ingest data into vector store
    """

    def __init__(
        self,
        raw_path: str,
        extracted_path: str,
        processed_path: str,
        chunks_path: str,
        vector_store_path: str,
    ):
        self.raw_path = Path(raw_path)
        self.extracted_path = Path(extracted_path)
        self.processed_path = Path(processed_path)
        self.chunks_path = Path(chunks_path)
        self.vector_store_path = vector_store_path

    def extract_data(self, verbose: bool = True) -> List[str]:
        """
        Extract data from raw files using Docling.
        """
        files = get_files_to_handle(self.raw_path, self.extracted_path, "pdf", "json")
        file_paths = [self.raw_path / file for file in files]
        output_paths = []

        if verbose:
            file_paths = tqdm(file_paths, desc="Extracting data", unit="file")
        else:
            file_paths = file_paths

        for file_path in file_paths:
            data = docling_pdf_to_json(file_path)
            output_path = self.extracted_path / f"{file_path.stem}.json"
            save_json_to_path(data=data, path=output_path)
            output_paths.append(output_path)
        return output_paths

    def process_data(self, verbose: bool = True) -> List[str]:
        """
        Process data by parsing Docling JSON.
        """
        files = get_files_to_handle(
            self.extracted_path, self.processed_path, "json", "json"
        )
        file_paths = [self.extracted_path / file for file in files]

        output_paths = []
        if verbose:
            file_paths = tqdm(file_paths, desc="Processing data", unit="file")
        else:
            file_paths = file_paths

        for file_path in file_paths:
            data = load_json_from_path(file_path)
            data = parse_docling_json(data)
            output_path = self.processed_path / f"{file_path.stem}.json"
            save_json_to_path(data=data, path=output_path)
            output_paths.append(output_path)
        return output_paths

    def chunk_data(self) -> None:
        """
        TODO: Chunk data
        """
        return print("Chunking data...")

    def ingest_data(self) -> None:
        """
        TODO: Ingest data into vector store
        """
        return print("Ingesting data into vector store...")

    def run(self) -> None:
        """
        Run ingestion pipeline to extract raw data, process data, create chunks, and ingest data into vector store
        """
        print("Running ingestion pipeline...")  # TODO: Add logging
        self.extract_data(verbose=True)
        self.process_data()
        print("Chunking data...")
        self.chunk_data()
        print("Ingesting data into vector store...")
        self.ingest_data()
        print("Ingestion pipeline completed...")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["raw_file_path"]
    extracted_path = config["extracted_file_path"]
    processed_path = config["processed_file_path"]
    chunks_path = config["chunks_file_path"]
    vector_store_path = config["vector_store_path"]

    ingester = Ingester(
        raw_path=raw_path,
        extracted_path=extracted_path,
        processed_path=processed_path,
        chunks_path=chunks_path,
        vector_store_path=vector_store_path,
    )
    ingester.run()
