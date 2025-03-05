"""Define the state structures for the agent."""

from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RankedDocument(BaseModel):
    """A ranked document."""

    document: str = Field(description="The document content")
    summary: str = Field(description="A summary of the document")
    score: int = Field(description="The score of the document between 0 and 100")


class RerankingState(BaseModel):
    """State for the reranking."""

    reranked_context: List[RankedDocument] = Field(description="The reranked context")


class State(TypedDict):
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    question: str
    context: List[Document]
    reranked_context: List[RankedDocument]
    answer: str
