from src.utils.base import BaseGenerator
from src.utils.schemas import RetrievedChunk
from typing import List, AsyncIterator
from src.online_components.retriever import SimpleHybridRetriever
from src.prompts.v1_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.online_components.reranker import SentenceTransformerReranker
import asyncio
from litellm import completion, acompletion
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict


class ChatMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def _trim(self):
        # keep last N turns (user+assistant pairs)
        excess = len(self.messages) - self.max_turns * 2
        if excess > 0:
            self.messages = self.messages[excess:]


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


_LLM_SEMAPHORE = asyncio.Semaphore(10)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _start_stream(**kwargs):
    """
    Retry ONLY the initial request that creates the stream.
    Do NOT retry once tokens begin flowing.
    """
    return await acompletion(stream=True, **kwargs)


async def stream_llm(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    history: list | None = None,
    temperature: float = 0.2,
    max_tokens: int = 300,
    timeout: int = 20,
) -> AsyncIterator[str]:
    if not user_prompt.strip():
        raise ValueError("Empty prompt")

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})

    async with _LLM_SEMAPHORE:
        try:
            stream = await _start_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.get("content")
                if delta:
                    yield delta

        except Exception as e:
            raise RuntimeError("LLM streaming request failed") from e


async def collect_stream(stream: AsyncIterator[str]) -> str:
    parts = []
    async for token in stream:
        parts.append(token)
    return "".join(parts)


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
        return answer


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
