import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
HYBRID_COLLECTION = os.getenv("HYBRID_COLLECTION_NAME")
DENSE_COLLECTION = os.getenv("DENSE_COLLECTION_NAME")
DENSE_EMBEDDING_MODEL = os.getenv("DENSE_MODEL")
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_MODEL")