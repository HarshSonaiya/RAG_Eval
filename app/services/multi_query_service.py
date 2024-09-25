from config.settings import settings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import MultiQueryRetriever
from config.logging_config import LoggerFactory  # Import LoggerFactory
from utils.llm_manager import LLMManager 


# Initialize logger using LoggerFactory
logger_factory = LoggerFactory()
logger = logger_factory.get_logger("pipeline")

class MultiQueryService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.vector_store = QdrantVectorStore(client=self.client, index_name=settings.DENSE_COLLECTION)
        self.retriever = self.vector_store.as_retriever(search_type="mmr")
        # Initialize ChatGroq as the LLM
        self.llm = LLMManager()
        
        # Create MultiQueryRetriever using the vector store and LLM
        self.multi_query_retriever = MultiQueryRetriever.from_llm(llm=self.llm, retriever=self.retriever)

    def multi_query_search(self, query: str, limit=5):
        """
        Perform a multi-query search based on the provided queries using MultiQueryRetriever.
        """
        # Use the generate method to retrieve results
        results = self.multi_query_retriever.invoke(query)
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLM based on the provided question and context.
        """
        response = self.llm.invoke({"question": question, "context": context})['text']
        return response
