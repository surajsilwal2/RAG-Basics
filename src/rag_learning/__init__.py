"""RAG learning package."""

from rag_learning.rag import (
    DEFAULT_K,
    DEFAULT_MAX_DISTANCE,
    NO_CONTEXT_ANSWER,
    GroundedAnswer,
    answer_from_context,
    build_prompt,
    format_context,
    retrieve,
    source_label,
)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_MAX_DISTANCE",
    "NO_CONTEXT_ANSWER",
    "GroundedAnswer",
    "answer_from_context",
    "build_prompt",
    "format_context",
    "retrieve",
    "source_label",
]


def main() -> None:
    print("Hello from rag-learning!")
