"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from agent.state import State

RFP_PATH = os.getcwd() + "/data/rfp.pdf"

loader = PyPDFLoader(RFP_PATH)

docs = loader.load()

embeddings = MistralAIEmbeddings(model="mistral-embed")

llm = ChatMistralAI(model="ministral-8b-latest", temperature=0, max_retries=2)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def retriever_node(state: State) -> State:
    """Retreive the answer to the question."""
    response = rag_chain.invoke({"input": state["question"]})
    return {"answer": response}


workflow = StateGraph(State)

workflow.add_node("retriever", retriever_node)
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", END)

graph = workflow.compile()

graph.name = "agent2"
