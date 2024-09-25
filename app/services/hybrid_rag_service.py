from config.settings import settings
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from tqdm import tqdm
from typing import List 
from config.logging_config import LoggerFactory  
from utils.llm_manager import LLMManager  


# Initialize logger using LoggerFactory
logger_factory = LoggerFactory()
logger = logger_factory.get_logger("pipeline")

class HybridRagService:
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
    
    def index_hybrid_collection(self, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant hybrid collection.
        """
        self.create_hybrid_collection()  # Create collection if it doesn't exist

        for i, doc in enumerate(tqdm(chunks, total=len(chunks))):
            dense_embedding = self.create_dense_vector(doc)
            sparse_embedding = self.create_sparse_vector(doc)

            self.client.upsert(
                collection_name=settings.HYBRID_COLLECTION,
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

    def create_hybrid_collection(self):
        """
        Create a hybrid collection in Qdrant if it does not exist.
        """
        if not self.client.collection_exists(collection_name=settings.HYBRID_COLLECTION):
            self.client.create_collection(
                collection_name=settings.HYBRID_COLLECTION,
                vectors_config={
                    'dense': models.VectorParams(size=384, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                }
            )
            logger.info(f"Created hybrid collection '{settings.HYBRID_COLLECTION}' in Qdrant.")

    def create_dense_vector(self, doc: Document):
        """
        Create a dense vector from the text using the dense embedding model.
        """
        embeddings = settings.DENSE_EMBEDDING_MODEL.encode(doc.page_content)
        return embeddings.tolist()

    def create_sparse_vector(self, doc: Document):
        """
        Create a sparse vector from the text using the sparse embedding model.
        """
        embeddings = list(settings.SPARSE_EMBEDDING_MODEL.embed([doc.page_content]))[0]
        return models.SparseVector(indices=embeddings.indices.tolist(), values=embeddings.values.tolist())

    def hybrid_search(self, query: str, limit=5):
        """
        Perform a hybrid search based on the provided query.
        """
        dense_query = list(settings.DENSE_EMBEDDING_MODEL.encode(query))
        sparse_query = list(settings.SPARSE_EMBEDDING_MODEL.embed([query]))[0]

        sparse_query = models.SparseVector(
            indices=sparse_query.indices.tolist(),
            values=sparse_query.values.tolist()
        )

        results = self.client.query_points(
            collection_name=settings.HYBRID_COLLECTION,
            prefetch=[
                models.Prefetch(query=sparse_query, using="sparse", limit=limit),
                models.Prefetch(query=dense_query, using="dense", limit=limit)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF)
        )

        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        response = self.llm_manager.llm.invoke({"question": question, "context": context})['text']
        return response
