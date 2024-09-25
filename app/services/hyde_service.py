from app.config.settings import settings
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
import logging
from app.utils.llm_manager import LLMManager
from app.utils.dense_collection import DenseCollection


logger = logging.getLogger("hyde_service")

class HyDEService:
    
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

    def hyde_search(self, query: str, limit=5):
        """
        Perform a HyDE search based on the provided query using QdrantVectorStore.
        """
        results = self.vector_store.as_retriever(query=query, limit=limit)
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        response = self.llm_manager.llm.invoke({"question": question, "context": context})['text']
        return response
