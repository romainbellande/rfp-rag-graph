"""Simple RAG agent using LlamaIndex."""

import asyncio
import os

from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from llama_index.core import Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from qdrant_client import AsyncQdrantClient, QdrantClient

from lindex.document_indexer import DocumentIndexer
from lindex.nodes import RetrieveNode
from lindex.state import State

model = ChatMistralAI(model="ministral-8b-latest", temperature=0, max_retries=2)
# creates a persistant index to disk
client = QdrantClient(url=os.getenv("QDRANT_URL"))
aclient = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
collection_name = "test"
Settings.embed_model = MistralAIEmbedding(
    model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY")
)
Settings.llm = MistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"), model="ministral-8b-latest"
)


RFP_PATH = os.getcwd() + "/data/rfp.pdf"

indexer = DocumentIndexer(
    pdf_path=RFP_PATH,
    collection_name=collection_name,
    recreate_collection=False,
)


# Initialize the indexer asynchronously
async def init_indexer():
    """Initialize the indexer asynchronously."""
    await indexer.ensure_collection()
    await indexer.init()


# Run the initialization in an event loop
asyncio.run(init_indexer())

workflow = StateGraph(State)

workflow.add_node("retrieve", RetrieveNode(document_indexer=indexer, model=model))
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", END)

graph = workflow.compile()

graph.name = "LlamaIndex Graph"
