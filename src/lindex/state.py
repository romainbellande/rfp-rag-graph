"""Define the state structures for the LlamaIndex."""

from typing import List, TypedDict

from llama_index.core import Document


class State(TypedDict):
    """State for the LlamaIndex."""

    question: str
    answer: str
    context: List[Document]
    # context: List[Document]
