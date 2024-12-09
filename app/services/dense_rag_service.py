import logging
from typing import List

from qdrant_client import QdrantClient, models
from utils.const import prompt_template
from utils.llm_manager import LLMManager

# Initialize logger using LoggerFactory
logger = logging.getLogger("pipeline")


class DenseRagService:
    def __init__(self, client: QdrantClient, llm_manager: LLMManager):
        self.client = client
        self.llm_manager = llm_manager
        self.prompt_template = prompt_template

    def dense_search(self, query: List[float], pdf_id: str, brain_id: str):
        """
        Perform a dense search based on the provided query using QdrantVectorStore.
        """
        logger.info("Begin dense Search")

        results = self.client.query_points(
            collection_name=brain_id,
            query=query,
            using="dense",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.pdf_id", match=models.MatchValue(value=pdf_id)
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
        formatted_prompt = self.prompt_template.format(
            question=question, context=context
        )

        response = self.llm_manager.llm.invoke(formatted_prompt)
        return response
