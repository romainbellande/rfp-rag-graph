"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

import os
from typing import Any, Dict, List, TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# from agent.state import State

llm = init_chat_model(
    "llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    model_provider="groq",
    temperature=0.3,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

client = QdrantClient(url=os.getenv("QDRANT_URL"))

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

RFP_PATH = os.getcwd() + "/data/rfp.pdf"


def embed_pdf(recreate=False):
    """Embed the PDF file."""
    if recreate:
        client.recreate_collection(
            collection_name="test",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    loader = PyPDFLoader(RFP_PATH)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)


# embed_pdf(True)

answer_generation_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.
You always respond in french.
Question: {question} 
Context: {context} 
Answer:
"""


class State(TypedDict):
    """State for the application."""

    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    """Retrieve the context for the question."""
    retrieved_docs = vector_store.similarity_search(state["question"], k=8)
    return {"context": retrieved_docs}


def generate(state: State):
    """Generate the answer for the question."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = PromptTemplate.from_template(answer_generation_template)
    messages = prompt.format(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"answer": response.content}


async def my_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    # configuration = Configuration.from_runnable_config(config)
    # configuration = Configuration.from_runnable_config(config)
    # You can use runtime configuration to alter the behavior of your
    # graph.

    return {"changeme": "output from my_node. Configured with"}


# Define a new graph
workflow = StateGraph(State)
workflow.add_sequence([retrieve, generate])

# Set the entrypoint as `call_model`
workflow.add_edge(START, "retrieve")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "RAG Graph"  # This defines the custom name in LangSmith
