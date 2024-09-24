from qdrant_client import QdrantClient, models
from app.config.logging_config import LoggerFactory
from app.config.settings import HYBRID_COLLECTION, QDRANT_URL

class QdrantClientManager:
    _client = None
    logger = LoggerFactory().get_logger("Qdrant")

    @classmethod
    def get_client(cls) -> QdrantClient:
        """
        Retrieve the singleton Qdrant client. If it doesn't exist, create it.

        Returns:
            QdrantClient: A single instance of QdrantClient.
        """
        if cls._client is None:
            cls._client = QdrantClient(url=QDRANT_URL)
            cls.logger.info("Qdrant client created.")
        return cls._client

    @classmethod
    def create_hybrid_collection(cls):
        """
        Create a collection in Qdrant if it does not exist.
        """
        client = cls.get_client()
        if not client.collection_exists(collection_name=HYBRID_COLLECTION):
            client.create_collection(
                collection_name=HYBRID_COLLECTION,
                vectors_config={
                    'dense': models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                }
            )
            cls.logger.info(f"Created hybrid collection '{HYBRID_COLLECTION}' in Qdrant.")
        else:
            cls.logger.info(f"Hybrid collection '{HYBRID_COLLECTION}' already exists.")
