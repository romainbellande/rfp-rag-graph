"""Rephrase the question."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent.state import State

rephrasing_template = """You are an assistant which rephrases questions in order to use in a RAG (Retrieval-Augmented Generation) pipeline.
You are given a question.
You need to rephrase the question to be more specific and to the point.
You must always keep the question in the same language as the question provided.

Question: {question} 
Answer:
"""


@dataclass
class RephraseNode:
    """Rephrase the question."""

    model: BaseChatModel

    def __call__(self, state: State) -> State:
        """Rephrase the question."""
        prompt = ChatPromptTemplate.from_template(rephrasing_template)
        messages = prompt.format(question=state["question"])
        response = self.model.invoke(messages)
        return {"question": response.content}
