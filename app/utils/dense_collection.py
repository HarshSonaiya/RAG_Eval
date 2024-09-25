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
    def __init__(self, client: QdrantClient, index_name: str):
        self.client = client
        self.index_name = index_name

    def create_dense_collection(self):
        """
        Create a dense collection in Qdrant if it does not exist.
        """
        if not self.client.collection_exists(collection_name=self.index_name):
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config={
                    'dense': models.VectorParams(size=384, distance=models.Distance.COSINE),
                }
            )
            logger.info(f"Created dense collection '{self.index_name}' in Qdrant.")

    def index_dense_collection(self, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant dense collection.
        """
        self.create_dense_collection()  # Create collection if it doesn't exist

        for i, doc in enumerate(tqdm(chunks, total=len(chunks))):
            dense_embedding = self.create_dense_vector(doc)

            self.client.upsert(
                collection_name=self.index_name,
                points=[models.PointStruct(
                    id=i,
                    vector={"dense": dense_embedding},
                    payload={
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )]
            )
        logger.info(f"Indexed {len(chunks)} documents into Qdrant Dense Collection.")

    def create_dense_vector(self, doc: Document):
        """
        Create a dense vector from the text using the dense embedding model.
        """
        embeddings = settings.DENSE_EMBEDDING_MODEL.encode(doc.page_content)
        return embeddings.tolist()
