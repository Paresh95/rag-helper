from utils.base import BaseGenerator
from utils.schemas import RetrievedChunk
from typing import List
from utils.prompt import PromptTemplate


class Generator(BaseGenerator):
    def __init__(self, model_name: str, prompt_template: PromptTemplate):
        self.model_name = model_name
        self.prompt_template = prompt_template

    def generate(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        prompt = self.prompt_template.format(
            query=query, retrieved_chunks=retrieved_chunks
        )
        return self.model.generate(prompt)
