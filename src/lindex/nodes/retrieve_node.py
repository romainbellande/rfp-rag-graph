"""Retrieve the documents."""

import pprint
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from lindex.document_indexer import DocumentIndexer
from lindex.state import State

answer_generation_template = """You are an assistant for question-answering tasks about RFP documents (Request for Proposals).
Use the provided search tool to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.
You always respond in french.
Only output the answer, no other text.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            answer_generation_template,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@dataclass
class RetrieveNode:
    """Retrieve the documents."""

    document_indexer: DocumentIndexer
    model: BaseChatModel

    def __call__(self, state: State) -> State:
        """Retrieve the documents."""
        search_tool = self.document_indexer.create_tool(
            name="search_tool",
            description=(
                "Provides information about rfp documents"
                "You can use this tool to search for documents by title, description, or content."
                "Use a detailed plain text question as input to the tool."
            ),
        )

        tools = [search_tool]

        agent = create_tool_calling_agent(
            llm=self.model,
            prompt=prompt,
            tools=tools,
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        response = agent_executor.invoke({"input": state["question"]})
        pprint.pprint(response)
        return {"answer": response["output"]}
