import asyncio
from litellm import completion, acompletion
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import AsyncIterator


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
