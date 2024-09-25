from qdrant_client import QdrantClient, models
from app.config.logging_config import LoggerFactory
from app.config.settings import settings

class QdrantClientManager:
    """Singleton manager for Qdrant client connections and collection management."""

    _instance = None

    def __new__(cls):
        """Ensure a single instance of QdrantClientManager (singleton)."""
        if cls._instance is None:
            cls._instance = super(QdrantClientManager, cls).__new__(cls)
            cls._instance.logger = LoggerFactory().get_logger("Qdrant")
            cls._instance.client = None  # Initialize client as None
        return cls._instance

    def get_client(self) -> QdrantClient:
        """
        Retrieve the singleton Qdrant client. If it doesn't exist, create it.

        Returns:
            QdrantClient: A single instance of QdrantClient.
        """
        if self.client is None:
            self.client = QdrantClient(url=settings.QDRANT_URL)
            self.logger.info("Qdrant client created.")
        return self.client

    def create_hybrid_collection(self) -> None:
        """
        Create a collection in Qdrant if it does not exist.
        """
        client = self.get_client()
        
        if not client.collection_exists(collection_name=settings.HYBRID_COLLECTION):
            self._create_collection(client)
            self.logger.info(f"Created hybrid collection '{settings.HYBRID_COLLECTION}' in Qdrant.")
        else:
            self.logger.info(f"Hybrid collection '{settings.HYBRID_COLLECTION}' already exists.")

    def _create_collection(self, client: QdrantClient) -> None:
        """
        Internal method to handle the creation of the hybrid collection.

        Args:
            client (QdrantClient): Qdrant client instance.
        """
        client.create_collection(
            collection_name=settings.HYBRID_COLLECTION,
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
