from app.config.settings import settings
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import MultiQueryRetriever
import logging
from typing import List 

logger = logging.getLogger("multi_query_service")

class MultiQueryService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.vector_store = QdrantVectorStore(client=self.client, index_name=settings.DENSE_COLLECTION)
        self.multi_query_retriever = MultiQueryRetriever(retriever=self.vector_store)

    def index_documents(self, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant collection.
        """
        # Here, you can call the existing indexing logic if needed
        # This can be left out if you don't want to duplicate indexing logic
        self.vector_store.index_documents(chunks)

    def multi_query_search(self, queries: List[str], limit=5):
        """
        Perform a multi-query search based on the provided queries using MultiQueryRetriever.
        """
        results = self.multi_query_retriever.retrieve(queries=queries, limit=limit)
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        response = self.llm_manager.llm.invoke({"question": question, "context": context})['text']
        return response
