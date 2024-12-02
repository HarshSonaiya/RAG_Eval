from config.settings import settings
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from utils.collection import Collection
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

    def dense_search(self, query: str,pdf_id: str, brain_id: str):
        """
        Perform a dense search based on the provided query using QdrantVectorStore.
        """
        collection_info = self.client.get_collection(collection_name=brain_id)
        logger.info("Collection info", collection_info)
        logger.info("Begin dense Search", len(query), query)

        results = self.client.query_points(
            collection_name=brain_id,
            query=query,  
            using = "dense",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.pdf_id",  
                        match=models.MatchValue(value=pdf_id)
                    )
                ]
            ),
        )
        documents = [point for point in results.points]
        logger.info(f"Dense Search Completed. {len(documents)}")
        return documents

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        formatted_prompt = self.prompt_template.format(question=question, context=context)

        response = self.llm_manager.llm.invoke(formatted_prompt)
        return response
