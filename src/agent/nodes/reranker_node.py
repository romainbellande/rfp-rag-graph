"""Rerank the documents."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent.state import RerankingState, State

reranking_template = """You are an assistant for question-answering tasks about RFP documents (Request for Proposals).
You are given a question and a list of documents.
You need to rerank the documents based on the relevance to the question.

Question: {question} 
Context: {context} 
Answer:
"""


@dataclass
class ReRankerNode:
    """Rerank the documents."""

    model: BaseChatModel

    def __call__(self, state: State) -> RerankingState:
        """Rerank the documents."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = ChatPromptTemplate.from_template(reranking_template)
        messages = prompt.format(question=state["question"], context=docs_content)
        response = self.model.with_structured_output(RerankingState).invoke(messages)
        sorted_docs = sorted(
            response.reranked_context, key=lambda x: x.score, reverse=True
        )
        top_5_docs = sorted_docs[:5]
        return {"reranked_context": top_5_docs}
