"""Retrieve the documents."""

from dataclasses import dataclass

from langchain_core.vectorstores import VectorStore

from agent.state import State


@dataclass
class RetrieveNode:
    """Retrieve the documents."""

    vector_store: VectorStore

    def __call__(self, state: State) -> State:
        """Retrieve the documents."""
        return {"context": self.vector_store.similarity_search(state["question"], k=15)}
