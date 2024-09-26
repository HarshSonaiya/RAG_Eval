from qdrant_client import QdrantClient, models
from langchain.schema import Document
from tqdm import tqdm
import logging
from config.settings import settings
from typing import List
from config.logging_config import LoggerFactory  


# Initialize logger using LoggerFactory
logger_factory = LoggerFactory()
logger = logger_factory.get_logger("pipeline")

class DenseCollection:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def create_dense_collection(self):
        """
        Create a dense collection in Qdrant if it does not exist.
        """

        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense":models.VectorParams(size=768, distance=models.Distance.COSINE),
                    }
            )
            logger.info(f"Created dense collection '{self.collection_name}' in Qdrant.")

