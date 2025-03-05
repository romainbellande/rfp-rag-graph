"""Generate the answer."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent.state import State

answer_generation_template = """You are an assistant for question-answering tasks about RFP documents (Request for Proposals).
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.
The context is in french.
You always respond in french.

Before answering, analyse each document in the context and identify  if it contains the answer to the question.
Assign a score between 0 and 100 to each document based on the relevance of the document to the question and then use this information to ignore documents that are not relevant.
Also, make sure to list the most relevant documents first and then answer the question based on the most relevant documents only.

Question: {question} 
Context: {context} 
Answer:
"""


@dataclass
class GenerateNode:
    """Generate the answer."""

    model: BaseChatModel

    def __call__(self, state: State) -> State:
        """Generate the answer for the question."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = ChatPromptTemplate.from_template(answer_generation_template)
        messages = prompt.format(question=state["question"], context=docs_content)
        response = self.model.invoke(messages)

        # Extract thoughts and answer from the response
        return {"answer": response.content}
