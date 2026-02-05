# TODO: Tidy up
import re
import tiktoken
from typing import Any, Dict, Iterable, List
from src.utils.schemas import Chunk

# --- Sentence splitting (simple, dependency-free) ---
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.
    If you need higher accuracy (abbreviations, etc.), swap this out for spacy/nltk.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# --- Token helpers ---
def get_encoder(model: str = "gpt-4o-mini"):
    # Use encoding_for_model when possible; fall back to a safe default.
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(enc, s: str) -> int:
    return len(enc.encode(s))


# --- Sentence-pack chunker using token budget ---
def pack_sentences_into_chunks(
    sentences: List[str],
    enc,
    max_tokens: int,
    overlap_sentences: int = 0,
) -> List[str]:
    """
    Packs sentences into chunk strings up to max_tokens.
    Optional overlap is in *sentences* (not tokens) between consecutive chunks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(enc, sent)

        # If a single sentence exceeds budget, flush current, then hard-split sentence by tokens.
        if sent_tokens > max_tokens:
            if cur:
                chunks.append(" ".join(cur).strip())
                cur, cur_tokens = [], 0

            # Hard split long sentence into token windows
            toks = enc.encode(sent)
            for i in range(0, len(toks), max_tokens):
                sub = enc.decode(toks[i : i + max_tokens]).strip()  # noqa: E203
                if sub:
                    chunks.append(sub)
            continue

        # If adding would exceed budget, emit current chunk.
        if (
            cur and (cur_tokens + sent_tokens + 1) > max_tokens
        ):  # +1 for a joining space-ish
            chunks.append(" ".join(cur).strip())

            # start next with overlap
            if overlap_sentences > 0:
                overlap = cur[-overlap_sentences:]
                cur = overlap[:]
                cur_tokens = count_tokens(enc, " ".join(cur))
            else:
                cur, cur_tokens = [], 0

        # add sentence
        cur.append(sent)
        cur_tokens += sent_tokens + 1

    if cur:
        chunks.append(" ".join(cur).strip())

    return chunks


# --- Main conversion: objects -> List[Chunk] ---
def objects_to_chunks(
    objs: Iterable[Dict[str, Any]],
    file_path: str,
    text_field: str = "text",
    model: str = "gpt-4o-mini",
    max_tokens: int = 300,
    overlap_sentences: int = 0,
    reset_index_per_object: bool = True,
) -> List[Chunk]:
    """
    Converts your input objects into Chunk objects.
    - chunk_index increments either per object (default) or globally.
    - tokens stored is the token count of the produced chunk string.
    """
    enc = get_encoder(model)
    out: List[Chunk] = []

    global_index = 0

    for obj in objs:
        text = (obj.get(text_field) or "").strip()
        if not text:
            continue

        sentences = split_sentences(text)
        chunk_strs = pack_sentences_into_chunks(
            sentences=sentences,
            enc=enc,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
        )

        local_index = 0
        for chunk_text in chunk_strs:
            idx = local_index if reset_index_per_object else global_index
            tok = count_tokens(enc, chunk_text)

            out.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "uuid": obj.get("uuid"),
                        "file_path": str(file_path),
                        "chunk_index": idx,
                        "section_path": obj.get("section_path"),
                        "section_refs": obj.get("section_refs"),
                        "page_no": obj.get("page_no"),
                        "tokens": tok,
                    },
                )
            )

            local_index += 1
            global_index += 1

    return out
