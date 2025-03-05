"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

import os
import pprint

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from agent.nodes import (
    GenerateNode,
    RephraseNode,
    ReRankerNode,
    RetrieveNode,
)
from agent.state import State

embeddings = MistralAIEmbeddings(model="mistral-embed")

model = ChatMistralAI(model="ministral-8b-latest", temperature=0, max_retries=2)

client = QdrantClient(url=os.getenv("QDRANT_URL"))

RFP_PATH = os.getcwd() + "/data/rfp.pdf"


def setup_vector_store(recreate=False):
    """Set up the vector store."""
    vectors_config = VectorParams(size=1024, distance=Distance.COSINE)
    if recreate:
        client.recreate_collection(
            collection_name="test",
            vectors_config=vectors_config,
        )
    if not client.collection_exists(collection_name="test"):
        pprint.pprint("Creating collection")
        client.create_collection(
            collection_name="test",
            vectors_config=vectors_config,
        )
    else:
        pprint.pprint("Collection already exists")


setup_vector_store()

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)


def load_pdf():
    """Load the PDF file."""
    loader = PDFPlumberLoader(RFP_PATH)
    docs = loader.load()

    return docs


def split_text(docs):
    """Split the text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    return text_splitter.split_documents(docs)


def index_docs(docs):
    """Index the documents."""
    vector_store.add_documents(docs)


def initial_setup():
    """Initialize the vector store with documents from the PDF."""
    # Load the PDF file if test collection is empty
    nb_docs = client.count(collection_name="test").count
    pprint.pprint(nb_docs)
    if nb_docs == 0:
        docs = load_pdf()

        all_splits = split_text(docs)

        index_docs(all_splits)


initial_setup()

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("retrieve", RetrieveNode(vector_store))
workflow.add_node("rerank", ReRankerNode(model))
workflow.add_node("generate", GenerateNode(model))
workflow.add_node("rephrase", RephraseNode(model))

workflow.add_edge(START, "rephrase")
workflow.add_edge("rephrase", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "RAG Graph"  # This defines the custom name in LangSmith
