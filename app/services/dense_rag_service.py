from config.settings import settings
from qdrant_client import QdrantClient
from langchain.schema import Document
from utils.dense_collection import DenseCollection
from langchain_qdrant import QdrantVectorStore
from config.logging_config import LoggerFactory  
from typing import List 
from utils.llm_manager import LLMManager
from uuid import uuid4



# Initialize logger using LoggerFactory
logger_factory = LoggerFactory()
logger = logger_factory.get_logger("pipeline")

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
        self.collection.create_dense_collection()
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=settings.DENSE_COLLECTION, 
            embedding = settings.DENSE_EMBEDDING_MODEL,
            vector_name= "dense",
        )
        self.retriever = self.vector_store.as_retriever(search_type="mmr")

    async def index_collection(self, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant dense collection.
        """
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        self.vector_store.add_documents(documents=chunks, ids= uuids)

    def dense_search(self, query: str):
        """
        Perform a dense search based on the provided query using QdrantVectorStore.
        """
        results = self.retriever.invoke(query)
        logger.info(f"Dense Search Completed. {results}")
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        formatted_prompt = self.prompt_template.format(question=question, context=context)

        response = self.llm_manager.llm.invoke(formatted_prompt)
        return response
