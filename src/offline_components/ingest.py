from typing import List
import json
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
from src.offline_components.utils.chunkers import objects_to_chunks
from src.offline_components.utils.upload_to_vector_store import upload_to_qdrant
from src.utils.vector_store import get_qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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
        collection_name: str,
        embedding_model_name: str,
    ):
        self.raw_path = Path(raw_path)
        self.extracted_path = Path(extracted_path)
        self.processed_path = Path(processed_path)
        self.chunks_path = Path(chunks_path)
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

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

    def chunk_data(self, verbose: bool = True) -> List[str]:
        """
        Chunk data into smaller chunks.
        """
        files = get_files_to_handle(
            self.processed_path, self.chunks_path, "json", "json"
        )
        file_paths = [self.processed_path / file for file in files]

        output_paths = []
        if verbose:
            file_paths = tqdm(file_paths, desc="Chunking data", unit="file")
        else:
            file_paths = file_paths

        for file_path in file_paths:
            data = load_json_from_path(file_path)
            data = objects_to_chunks(
                data,
                file_path=file_path,
                text_field="text",
                model="gpt-4o-mini",
                max_tokens=300,
                overlap_sentences=0,
            )
            output_path = self.chunks_path / f"{file_path.stem}.json"
            data_to_save = [
                c.model_dump() for c in data
            ]  # Convert Pydantic objects to dictionaries
            save_json_to_path(data=data_to_save, path=output_path)
            output_paths.append(output_path)
        return output_paths

    def ingest_data(self, verbose: bool = True) -> None:
        """
        Ingest data into vector store.
        """
        client = get_qdrant_client(url=self.vector_store_path)
        embedding_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        vector_store = QdrantVectorStore(
            client=client,
            embedding_model=embedding_model,
            collection_name=self.collection_name,
        )

        files = Path(self.chunks_path).glob("*.json")
        if verbose:
            files = tqdm(files, desc="Ingesting data", unit="file")
        else:
            files = files
        for file in files:
            try:
                data = load_json_from_path(file)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON file: {file} ({exc})")
                continue
            upload_to_qdrant(
                chunks=data,
                vector_store=vector_store,
                embedding_model=embedding_model,
            )
        return True

    def run(self) -> None:
        """
        Run ingestion pipeline to extract raw data, process data, create chunks, and ingest data into vector store
        """
        print("Running ingestion pipeline...")  # TODO: Add logging
        self.extract_data(verbose=True)
        self.process_data()
        self.chunk_data()
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
    collection_name = config["collection_name"]
    embedding_model_name = config["embedding_model_name"]

    ingester = Ingester(
        raw_path=raw_path,
        extracted_path=extracted_path,
        processed_path=processed_path,
        chunks_path=chunks_path,
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )
    ingester.run()
