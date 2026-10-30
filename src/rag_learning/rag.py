"""Grounded RAG: answer only from the retrieved passages.

The default LangChain pattern — put `context` in a prompt and hope — leaves two
ways for the model to answer from its own training data instead of your index:

1. **Weak retrieval leaks in.** If the passages are irrelevant but still present,
   the model treats them as a formality and answers from memory.
2. **The prompt permits it.** "You can answer using this context" is an option,
   not a constraint. With an empty or useless context the model simply continues
   from what it already knows.

This module closes both:

* :func:`retrieve` deduplicates and drops passages above a distance cutoff, so
  irrelevant text never reaches the prompt as context.
* :func:`answer_from_context` refuses without calling the model at all when
  nothing relevant was retrieved, and otherwise sends a prompt that forbids
  outside knowledge and asks for inline citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from langchain_core.documents import Document

#: Number of passages handed to the model.
DEFAULT_K = 4

#: Maximum squared-L2 distance (Chroma's default space) a passage may have and
#: still count as relevant. Distances here are `2 - 2·cosine`, so `1.25` is a
#: cosine similarity of about `0.38`.
#:
#: Measured on this repository's `Vector/` store with `all-MiniLM-L6-v2`:
#: questions the PDF answers score `0.62`–`1.14`; questions it cannot answer
#: ("capital of France", "replace my laptop battery", "sourdough recipe") score
#: `1.58`–`1.75`. The cutoff sits in the empty band between them.
DEFAULT_MAX_DISTANCE = 1.25

#: Returned verbatim when retrieval finds nothing relevant. No model call is
#: made, which is the only way to be certain the answer is not from memory.
NO_CONTEXT_ANSWER = (
    "The indexed documents do not cover this. No retrieved passage was close "
    "enough to the question to answer from."
)

GROUNDING_PROMPT = """You answer questions from a document index. Use ONLY the numbered passages below.

Rules:
- Every statement must be supported by a passage.
- Do not use your own knowledge, do not guess, and do not fill gaps from memory.
- Cite the passage you used inline, e.g. [1].
- If the passages do not contain the answer, reply with exactly: {refusal}

Passages:
{context}

Question: {question}

Answer:"""


class VectorStore(Protocol):
    """The slice of the Chroma interface this module needs."""

    def similarity_search_with_score(
        self, query: str, k: int = ...
    ) -> Sequence[tuple[Document, float]]: ...


class ChatModel(Protocol):
    """The slice of the LangChain chat-model interface this module needs."""

    def invoke(self, input: Any) -> Any: ...


@dataclass(frozen=True)
class GroundedAnswer:
    """An answer plus the evidence it was allowed to use."""

    answer: str
    passages: tuple[Document, ...] = field(default_factory=tuple)
    prompt: str | None = None

    @property
    def used_retrieval(self) -> bool:
        """False when the pipeline refused before reaching the model."""
        return bool(self.passages)

    @property
    def sources(self) -> tuple[str, ...]:
        """Human-readable labels for the passages, in citation order."""
        return tuple(source_label(passage) for passage in self.passages)

    def __str__(self) -> str:
        return self.answer


def _page_number(document: Document) -> str | None:
    """Best-effort page for a passage, from the loader's metadata."""
    metadata = document.metadata or {}
    if metadata.get("page_label") is not None:
        return str(metadata["page_label"])
    if metadata.get("page") is not None:
        # PyPDFLoader's `page` is 0-based; page_label is what a reader sees.
        return str(int(metadata["page"]) + 1)
    return None


def _file_name(document: Document) -> str:
    """`fabric-onelake.pdf` — the metadata source without its directory."""
    metadata = document.metadata or {}
    return str(metadata.get("source") or "unknown").replace("\\", "/").rsplit("/", 1)[-1]


def source_label(document: Document) -> str:
    """`fabric-onelake.pdf, page 3` — the attribution shown to the user."""
    page = _page_number(document)
    return f"{_file_name(document)}, page {page}" if page else _file_name(document)


def _dedupe_key(document: Document) -> tuple[str, str]:
    """Identity of a passage, so the same text is never sent twice.

    Duplicates arrive two ways: the same page ingested twice (same text, same
    page, different vector ids), and the same snippet reprinted on two pages.
    The document ID cannot be used because it differs in both cases, so the key
    is the file plus the normalised text. The page is deliberately excluded -
    otherwise a reprinted snippet still occupies two of the `k` slots.
    """
    return _file_name(document), " ".join(document.page_content.split())


def retrieve(
    store: VectorStore,
    question: str,
    *,
    k: int = DEFAULT_K,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> list[Document]:
    """Return the top `k` *relevant, distinct* passages for `question`.

    Over-fetches, then drops passages above `max_distance` and repeats of the
    same text, so `k` slots are spent on distinct evidence rather than on the
    same page twice.
    """
    if k <= 0:
        return []

    hits = store.similarity_search_with_score(question, k=max(k * 4, k))
    passages: list[Document] = []
    seen: set[tuple[str, str]] = set()

    for document, distance in hits:
        if max_distance is not None and distance > max_distance:
            continue
        key = _dedupe_key(document)
        if key in seen:
            continue
        seen.add(key)
        passages.append(document)
        if len(passages) == k:
            break

    return passages

def format_context(passages: Sequence[Document]) -> str:
    """Render passages as numbered blocks the model can cite by index."""
    blocks = []
    for position, document in enumerate(passages, start=1):
        # PDF text carries hard line breaks; collapse them so the model is not
        # misled into treating a wrapped sentence as two separate facts.
        text = " ".join(document.page_content.split())
        blocks.append(f"[{position}] {source_label(document)}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, passages: Sequence[Document]) -> str:
    """The prompt actually sent, built from the grounding template."""
    return GROUNDING_PROMPT.format(
        refusal=NO_CONTEXT_ANSWER,
        context=format_context(passages),
        question=question,
    )


def answer_from_context(
    store: VectorStore,
    llm: ChatModel,
    question: str,
    *,
    k: int = DEFAULT_K,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> GroundedAnswer:
    """Answer `question` from the index, or refuse without asking the model."""
    passages = retrieve(store, question, k=k, max_distance=max_distance)

    if not passages:
        return GroundedAnswer(answer=NO_CONTEXT_ANSWER)

    prompt = build_prompt(question, passages)
    reply = llm.invoke(prompt)
    text = getattr(reply, "content", reply)
    return GroundedAnswer(
        answer=str(text).strip(),
        passages=tuple(passages),
        prompt=prompt,
    )
