"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

import os
import pprint
from typing import List, TypedDict

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

llm = OllamaLLM(model="deepseek-r1:14b")
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
client = QdrantClient(url=os.getenv("QDRANT_URL"))


RFP_PATH = os.getcwd() + "/data/rfp.pdf"


class DeepSeekOutput(BaseModel):
    """Output for the DeepSeek model."""

    thoughts: str = Field(description="Your thoughts on the question")
    answer: str = Field(description="The answer to the question")


def setup_vector_store(recreate=False):
    """Set up the vector store."""
    vectors_config = VectorParams(size=5120, distance=Distance.COSINE)
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


# embed_pdf(True)

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


class State(TypedDict):
    """State for the application."""

    question: str
    context: List[Document]
    answer: str
    thoughts: str


def retrieve(state: State):
    """Retrieve the context for the question."""
    retrieved_docs = vector_store.similarity_search(state["question"], k=10)
    return {"context": retrieved_docs}


def generate(state: State):
    """Generate the answer for the question."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = ChatPromptTemplate.from_template(answer_generation_template)
    messages = prompt.format(question=state["question"], context=docs_content)
    response = llm.invoke(messages)

    # Extract thoughts and answer from the response
    return extract_thoughts_and_answer(response)


def extract_thoughts_and_answer(response):
    """Extract thoughts and answer from the response."""
    response_text = str(response)

    # Extract thoughts if they exist between <think> tags
    thoughts = ""
    if "<think>" in response_text and "</think>" in response_text:
        start_idx = response_text.find("<think>") + len("<think>")
        end_idx = response_text.find("</think>")
        thoughts = response_text[start_idx:end_idx].strip()

    # The answer is everything after </think> or the whole response if no think tags
    if "</think>" in response_text:
        answer = response_text[
            response_text.find("</think>") + len("</think>") :
        ].strip()
    else:
        answer = response_text

    pprint.pprint(response)
    return {"answer": answer, "thoughts": thoughts}


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
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Set the entrypoint as `call_model`
workflow.add_edge(START, "retrieve")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "RAG Graph"  # This defines the custom name in LangSmith
