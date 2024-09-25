from app.config.settings import settings
from qdrant_client import QdrantClient
from langchain.schema import Document
from app.utils.dense_collection import DenseCollection
from langchain_qdrant import QdrantVectorStore
import logging
from typing import List 
from app.utils.llm_manager import LLMManager


logger = logging.getLogger("dense_rag_service")

class DenseRagService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.llm_manager = LLMManager()  
        self.prompt_template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown: """
        self.collection = DenseCollection(client, settings.DENSE_COLLECTION)
        self.vector_store = QdrantVectorStore(client=self.client, index_name=settings.DENSE_COLLECTION)

    def index_collection(self, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant dense collection.
        """
        self.collection.index_dense_collection(chunks)

    def dense_search(self, query: str, limit=5):
        """
        Perform a dense search based on the provided query using QdrantVectorStore.
        """
        results = self.vector_store.as_retriever(query=query, limit=limit)

        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        response = self.llm_manager.llm.invoke({"question": question, "context": context})['text']
        return response
