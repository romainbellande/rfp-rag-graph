"""Graph for the LlamaIndex."""

import logging
import os

from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.extractors import (
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from lindex2.state import State

model = ChatMistralAI(model="ministral-8b-latest", temperature=0, max_retries=2)

Settings.embed_model = MistralAIEmbedding(
    model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"), embed_batch_size=50
)

llm = MistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="ministral-8b-latest",
    temperature=0,
    max_tokens=1024,
)

Settings.llm = llm

RFP_PATH = os.getcwd() + "/data/rfp.pdf"

client = QdrantClient(url=os.getenv("QDRANT_URL"))
aclient = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
collection_name = "lindex2"

vector_store = QdrantVectorStore(
    collection_name=collection_name,
    client=client,
    batch_size=40,
)

logging.info("Creating storage context")

# storage_context = StorageContext.from_defaults(vector_store=vector_store)

logging.info("Creating transformations")

transformations = [
    TokenTextSplitter(chunk_size=512, chunk_overlap=128, separator=" "),
    TitleExtractor(),
]

logging.info("Creating cache")

redis_cache = RedisCache.from_host_and_port(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
)

logging.info("Creating pipeline")

pipeline = IngestionPipeline(
    # cache=IngestionCache(
    #     cache=redis_cache,
    # ),
    vector_store=vector_store,
    transformations=transformations,
)

logging.info("Creating loader")

loader = SmartPDFLoader(llmsherpa_api_url=os.getenv("LLMSHERPA_API_URL"))

documents = loader.load_data(RFP_PATH)

logging.info("Running pipeline")

pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=Settings.embed_model,
)

logging.info("Creating query engine tool")

rfp_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    response_mode="tree_summarize",
    verbose=True,
)

query_engine_tool = QueryEngineTool(
    query_engine=rfp_engine,
    metadata=ToolMetadata(
        name="rfp_query_engine",
        description=(
            "This tool searches and extracts information from the RFP document. "
            "Input should be a specific question or search query about the RFP. "
            "The tool will return relevant information from the document. "
            "Always use this tool to find information before answering any question."
        ),
    ),
)

logging.info("Creating agent")

agent = FunctionCallingAgent.from_tools(
    [query_engine_tool],
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are an assistant specialized in analyzing RFP documents. Follow these rules:\n"
        "1. ALWAYS use the rfp_query_engine tool first to search for information before answering.\n"
        "2. If you need to find specific information, make your search query clear and focused.\n"
        "3. Base your answers ONLY on information returned by the tool.\n"
        "4. If the tool doesn't return relevant information, try reformulating your search query.\n"
        "5. If you still cannot find the information after trying, explain what you searched for and that it wasn't found.\n"
    ),
)


def retreiver(state: State) -> State:
    """Retreive the answer to the question."""
    # Ensure the question is properly formatted for the agent
    formatted_question = (
        f"Please search and provide information about: {state['question']}"
    )
    # response = agent.chat(formatted_question)
    response = rfp_engine.query(formatted_question)
    logging.info("Response:")
    logging.info(response)
    logging.info("End of response")
    return {"answer": response.response}


workflow = StateGraph(State)

workflow.add_node("retreiver", retreiver)
workflow.add_edge(START, "retreiver")
workflow.add_edge("retreiver", END)


graph = workflow.compile()

graph.name = "lindex2"
