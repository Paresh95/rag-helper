from utils.base import BaseIngester
from utils.schemas import Document
from typing import List


class Ingester(BaseIngester):
    """
    TODO: ADD DESCRIPTION
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def ingest(self, documents: List[Document]) -> None:
        pass
