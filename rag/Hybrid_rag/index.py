from app.config.settings import HYBRID_COLLECTION, DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL, DENSE_COLLECTION
from qdrant_client import QdrantClient
from tqdm import tqdm
from langchain.schema import Document
from qdrant_client import models
from typing import List
from app.config.logging_config import LoggerFactory

logger = LoggerFactory()
logger = logger.get_logger("pipeline")


class Indexer:
    @staticmethod
    def index_hybrid_collection(client: QdrantClient, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant hybrid collection.

        Args:
            client
            chunks (List[Document]): A list of Document objects to be indexed.
        """
        Indexer.create_hybrid_collection(client)  # Create collection if it doesn't exist

        for i, doc in enumerate(tqdm(chunks, total=len(chunks))):
            dense_embedding = Indexer.create_dense_vector(doc)
            sparse_embedding = Indexer.create_sparse_vector(doc)

            client.upsert(
                collection_name=HYBRID_COLLECTION,
                points=[models.PointStruct(
                    id=i,
                    vector={
                        "dense": dense_embedding,
                        "sparse": sparse_embedding
                    },
                    payload={
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )]
            )
        logger.info(f"Indexed {len(chunks)} documents into Qdrant Hybrid Collection.")

    @staticmethod
    def create_sparse_vector(doc: Document):
        """
        Create a sparse vector from the text using the sparse embedding model.

        Args:
            doc (Document): A Document object with 'page_content'.

        Returns:
            SparseVector: A Qdrant SparseVector object.
        """
        embeddings = list(SPARSE_EMBEDDING_MODEL.embed([doc.page_content]))[0]

        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            return models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")

    @staticmethod
    def create_dense_vector(doc: Document):
        """
        Create a dense vector from the text using the dense embedding model.

        Args:
            doc (Document): A Document object with 'page_content'.

        Returns:
            List[float]: A list of embeddings for the document.
        """
        embeddings = DENSE_EMBEDDING_MODEL.encode(doc.page_content)
        return embeddings.tolist()

    @staticmethod
    def create_hybrid_collection(client):
        """
        Create a collection in Qdrant if it does not exist.

        Args:
            client (QdrantClient): An instance of QdrantClient.
        """
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
            logger.info(f"Created hybrid collection '{HYBRID_COLLECTION}' in Qdrant.")

    @staticmethod
    def create_dense_collection(client):
        """
        Create a collection in Qdrant if it does not exist.

        Args:
            client (QdrantClient): An instance of QdrantClient.
        """
        if not client.collection_exists(collection_name=DENSE_COLLECTION):
            client.create_collection(
                collection_name=DENSE_COLLECTION,
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
            logger.info(f"Created hybrid collection '{HYBRID_COLLECTION}' in Qdrant.")

    @staticmethod
    def index_dense_collection(client: QdrantClient, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant hybrid collection.

        Args:
            client
            chunks (List[Document]): A list of Document objects to be indexed.
        """
        Indexer.create_dense_collection(client)  # Create collection if it doesn't exist

        for i, doc in enumerate(tqdm(chunks, total=len(chunks))):
            dense_embedding = Indexer.create_dense_vector(doc)

            client.upsert(
                collection_name=HYBRID_COLLECTION,
                points=[models.PointStruct(
                    id=i,
                    vector={
                        "dense": dense_embedding,
                    },
                    payload={
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )]
            )
        logger.info(f"Indexed {len(chunks)} documents into Qdrant Dense Collection.")