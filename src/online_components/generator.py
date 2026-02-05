from src.utils.base import BaseGenerator
from src.utils.schemas import RetrievedChunk
from typing import List
from src.utils.prompt import PromptTemplate
from litellm import completion
from src.online_components.retriever import SimpleHybridRetriever
from src.prompts.v1_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.online_components.reranker import SentenceTransformerReranker


def format_context(nodes: List[RetrievedChunk]) -> str:
    formatted = []
    for i, node in enumerate(nodes, start=1):
        formatted.append(f"""Source {i}: {node.text}""")
    return "\n\n".join(formatted)


def call_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> str:
    response = completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


class Generator(BaseGenerator):
    def __init__(self, model_name: str, prompt_template: PromptTemplate):
        self.model_name = model_name
        self.prompt_template = prompt_template

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt_template: str = USER_PROMPT_TEMPLATE,
    ) -> str:
        context = format_context(retrieved_chunks)

        user_prompt = user_prompt_template.format(
            context=context,
            question=question,
        )
        response = call_llm(self.model_name, system_prompt, user_prompt)
        return response


if __name__ == "__main__":
    retriever = SimpleHybridRetriever(
        similarity_top_k=10, sparse_top_k=10, hybrid_top_k=7
    )
    retriever_query = (
        "As shown in figure 2, the semantic chunking algorithm works by first splitting"
    )
    generator_query = "What does figure 2 show?"
    retrieved_nodes = retriever.retrieve(retriever_query)
    reranked_nodes = SentenceTransformerReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3
    ).rerank(retriever_query, retrieved_nodes)
    for node in reranked_nodes:
        print(node.score)
        print(node.node.metadata["similarity_score"])
        print(node.text)
        print("-" * 50)
    answer = Generator(
        model_name="gpt-4o-mini", prompt_template=USER_PROMPT_TEMPLATE
    ).generate(generator_query, reranked_nodes)
    print(answer)
