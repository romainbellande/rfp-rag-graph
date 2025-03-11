from typing import List, TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    """State for the agent."""

    question: str
    context: List[Document]
    answer: str
