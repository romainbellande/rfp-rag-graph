"""Simple RAG agent using LlamaIndex."""

import asyncio
import logging
import os
from typing import List

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.extractors import (
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

# creates a persistant index to disk
client = QdrantClient(url=os.getenv("QDRANT_URL"))
aclient = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

Settings.embed_model = MistralAIEmbedding(
    model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY")
)

Settings.llm = MistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="ministral-8b-latest",
    temperature=0,
    max_retries=10,
)


transformations = [
    TokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    ),
    TitleExtractor(),
    # QuestionsAnsweredExtractor(questions=3),
]


class DocumentIndexer:
    """Document indexer."""

    client: QdrantClient
    aclient: AsyncQdrantClient
    vector_store: QdrantVectorStore
    pipeline: IngestionPipeline
    pdf_path: str
    collection_name: str
    storage_context: StorageContext
    recreate_collection: bool

    def __init__(
        self, pdf_path: str, collection_name: str, recreate_collection: bool = False
    ):
        """Initialize the document indexer."""
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"))
        self.aclient = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
        self.collection_name = collection_name
        self.recreate_collection = recreate_collection
        self.pdf_path = pdf_path

    async def ensure_collection(self, recreate=False):
        """Ensure the collection exists."""
        logging.info(f"Ensuring collection {self.collection_name} exists")

        vectors_config = {
            "text-dense": VectorParams(size=1024, distance=Distance.COSINE),
        }

        sparse_vectors_config = {
            "text-sparse": SparseVectorParams(),
        }

        if recreate:
            logging.info(f"Recreating collection {self.collection_name}")
            await self.aclient.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        if not await self.aclient.collection_exists(
            collection_name=self.collection_name
        ):
            logging.info(f"Creating collection {self.collection_name}")
            await self.aclient.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

    async def load_documents(self):
        """Load the documents from the PDF."""
        logging.info(f"Loading documents from {self.pdf_path}")
        loader = SmartPDFLoader(llmsherpa_api_url=os.getenv("LLMSHERPA_API_URL"))
        documents = await asyncio.to_thread(loader.load_data, self.pdf_path)
        logging.info(f"Loaded {len(documents)} documents")
        return documents

    def embed_node(self, node):
        """Embed a single node using the embedding model."""
        if not node.embedding:
            content = node.get_content(metadata_mode="all")
            embedding = Settings.embed_model.get_text_embedding(content)
            node.embedding = embedding
        return node

    async def ingest_documents(self, documents: List[Document]):
        """Ingest the documents into the vector store."""
        logging.info(f"Ingesting {len(documents)} documents into the vector store")
        nodes = await self.pipeline.arun(
            documents=documents,
            in_place=True,
            show_progress=True,
        )

        # # Embed nodes before adding to vector store
        # logging.info("Embedding nodes...")
        # embed_model = Settings.embed_model

        # # Process embeddings sequentially to avoid rate limits
        processed_nodes = []

        for node in nodes:
            processed_node = self.embed_node(node)
            processed_nodes.append(processed_node)

        nodes = processed_nodes

        self.vector_store.add(nodes)
        logging.info(f"Added {len(nodes)} nodes to the vector store")

    def create_tool(self, name: str, description: str):
        """Create a tool."""
        logging.info(f"Creating tool {name}")
        return QueryEngineTool(
            query_engine=self.index.as_query_engine(),
            metadata=ToolMetadata(
                name=name,
                description=description,
            ),
        ).to_langchain_tool()

    async def init(self):
        """Initialize the vector store with documents from the PDF."""
        # Load the PDF file if test collection is empty
        await self.ensure_collection(self.recreate_collection)

        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.client,
            aclient=self.aclient,
            batch_size=20,
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        self.pipeline = IngestionPipeline(
            transformations=transformations,
            cache=IngestionCache(
                cache=RedisCache.from_host_and_port(
                    host=os.getenv("REDIS_HOST"),
                    port=os.getenv("REDIS_PORT"),
                )
            ),
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=Settings.embed_model
        )
        logging.info(f"Counting documents in collection {self.collection_name}")
        nb_docs = await self.aclient.count(collection_name=self.collection_name)
        count = nb_docs.count

        logging.info(
            f"Number of documents in collection {self.collection_name}: {count}"
        )

        if count == 0:
            documents = await self.load_documents()
            await self.ingest_documents(documents)
