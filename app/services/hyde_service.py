from config.settings import settings
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from config.logging_config import LoggerFactory  
from utils.llm_manager import LLMManager
from utils.dense_collection import DenseCollection


# Initialize logger using LoggerFactory
logger_factory = LoggerFactory()
logger = logger_factory.get_logger("pipeline")

class HyDEService:
    
    def __init__(self, client: QdrantClient):
        self.client = client
        self.llm_manager = LLMManager()  
        self.prompt_template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question.If you are not provided with any extracted
        parts of the documments then try to generate a hypothetical answer. Remember to provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown: """
        self.collection = DenseCollection(client, settings.DENSE_COLLECTION)
        self.vector_store = QdrantVectorStore(client=self.client, index_name=settings.DENSE_COLLECTION)
        self.retriever = self.vector_store.as_retriever(search_type="mmr")


    def hyde_search(self, query: str):
        """
        Perform a HyDE search based on the provided query using QdrantVectorStore.
        """
        results = self.retriever.invoke(query)
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        response = self.llm_manager.llm.invoke({"question": question, "context": context})['text']
        return response
