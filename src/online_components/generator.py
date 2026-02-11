import asyncio
from typing import List, AsyncIterator
from src.utils.base import BaseGenerator
from src.utils.schemas import RetrievedChunk
from src.online_components.retriever import SimpleHybridRetriever
from src.online_components.prompts.v1_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.online_components.reranker import SentenceTransformerReranker
from src.online_components.utils.memory import ChatMemory
from src.online_components.utils.llm import stream_llm, collect_stream
from src.online_components.utils.guardrails import validate_answer


def format_context(nodes: List[RetrievedChunk]) -> str:
    formatted = []
    for i, node in enumerate(nodes, start=1):
        formatted.append(f"""Source {i}: {node.text}""")
    return "\n\n".join(formatted)


class Generator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt_template: str,
        memory: ChatMemory | None = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.memory = memory or ChatMemory()

    async def stream(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> AsyncIterator[str]:
        context = format_context(retrieved_chunks)
        user_prompt = self.user_prompt_template.format(
            context=context, question=question
        )
        history = self.memory.get()
        stream = stream_llm(
            model=self.model_name,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            history=history,
        )

        self.memory.add_user(question)

        return stream

    async def generate(
        self, question: str, retrieved_chunks: List[RetrievedChunk]
    ) -> str:
        stream = await self.stream(question, retrieved_chunks)
        answer = await collect_stream(stream)
        self.memory.add_assistant(answer)
        validated_answer = validate_answer(answer)
        return validated_answer


async def main():
    memory = ChatMemory(max_turns=5)
    generator = Generator(
        model_name="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        memory=memory,
    )
    retriever = SimpleHybridRetriever(
        similarity_top_k=10, sparse_top_k=10, hybrid_top_k=7
    )

    retriever_query = (
        "As shown in figure 2, the semantic chunking algorithm works by first splitting"
    )

    retrieved_nodes = retriever.retrieve(retriever_query)

    reranked_nodes = SentenceTransformerReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3,
    ).rerank(retriever_query, retrieved_nodes)

    answer1 = await generator.generate("What does figure 2 show?", reranked_nodes)
    answer2 = await generator.generate("Why is that important?", reranked_nodes)

    print(answer1)
    print(answer2)


async def chat_loop():
    memory = ChatMemory(max_turns=5)

    generator = Generator(
        model_name="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        memory=memory,
    )

    retriever = SimpleHybridRetriever(
        similarity_top_k=10, sparse_top_k=10, hybrid_top_k=7
    )

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        retrieved = retriever.retrieve(question)
        reranked = SentenceTransformerReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=3,
        ).rerank(question, retrieved)

        print("Assistant: ", end="", flush=True)

        stream = await generator.stream(question, reranked)
        answer = await collect_stream(stream)

        generator.memory.add_assistant(answer)
        print(answer)


if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(chat_loop())
