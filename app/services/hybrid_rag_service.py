import logging
import re
import uuid
from typing import List

from config.settings import settings
from langchain.schema import Document
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from utils.const import prompt_template
from utils.llm_manager import LLMManager

# Initialize logger using LoggerFactory
logger = logging.getLogger("pipeline")


class HybridRagService:
    def __init__(self, client: QdrantClient, llm_manager: LLMManager):
        self.client = client
        self.llm_manager = llm_manager
        self.prompt_template = prompt_template

    async def index_hybrid_collection(self, chunks: List[Document], brain_id: str):
        """
        Index the given list of Document chunks into the Qdrant hybrid collection.
        """
        logger.info(f"Indexing {len(chunks)} documents into Qdrant Hybrid Collection.")

        try:
            invalid_chunks = 0
            for i, doc in enumerate(tqdm(chunks, total=len(chunks))):
                # Create embeddings with fallback logic
                try:
                    dense_embedding = self.create_dense_vector(doc.page_content)
                except Exception as e:
                    logger.exception(
                        f"Error creating dense vector for document {i}: {e}"
                    )
                    dense_embedding = None  # Fallback to None

                try:
                    sparse_embedding = self.create_sparse_vector(doc.page_content)
                except Exception as e:
                    logger.exception(
                        f"Error creating sparse vector for document {i}: {e}"
                    )
                    sparse_embedding = None  # Fallback to None

                # Only upsert if embeddings were created successfully
                if dense_embedding is not None and sparse_embedding is not None:
                    self.client.upsert(
                        collection_name=brain_id,
                        points=[
                            models.PointStruct(
                                id=str(uuid.uuid4()),
                                vector={
                                    "dense": dense_embedding,
                                    "sparse": sparse_embedding,
                                },
                                payload={
                                    "content": doc.page_content,
                                    "metadata": doc.metadata,
                                },
                            )
                        ],
                    )
                else:
                    logger.warning(
                        f"Skipping indexing for document {i} due to failed embeddings."
                    )
                    invalid_chunks += 1

            if invalid_chunks > 0:
                return False
            else:
                return True
        except Exception as e:
            logger.exception(f"Error processing document {i}: {e}")

        logger.info(f"Indexed {len(chunks)} documents into Qdrant Hybrid Collection.")

    def create_dense_vector(self, text: str):
        """
        Create a dense vector from the text using the dense embedding model.
        """
        try:
            embeddings = settings.DENSE_EMBEDDING_MODEL.embed_query(text)
            return embeddings
        except Exception as e:
            raise e

    def create_sparse_vector(self, text: str):
        """
        Create a sparse vector from the text using the sparse embedding model.
        """
        try:
            embeddings = list(settings.SPARSE_EMBEDDING_MODEL.embed([text]))[0]
            return models.SparseVector(
                indices=embeddings.indices.tolist(), values=embeddings.values.tolist()
            )
        except Exception as e:
            raise e

    def hybrid_search(self, query: str, selected_pdf_id: str, brain_id: str, limit=20):
        """
        Perform a hybrid search based on the provided query.
        """
        logger.info(f"Performing hybrid search for the selected pdf")

        dense_query = list(settings.DENSE_EMBEDDING_MODEL.embed_query(query))
        sparse_query = list(settings.SPARSE_EMBEDDING_MODEL.embed([query]))[0]

        sparse_query = models.SparseVector(
            indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()
        )

        results = self.client.query_points(
            collection_name=brain_id,
            prefetch=[
                models.Prefetch(query=sparse_query, using="sparse", limit=limit),
                models.Prefetch(query=dense_query, using="dense", limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.pdf_id",
                        match=models.MatchValue(value=selected_pdf_id),
                    )
                ]
            ),
            limit=limit,
        )

        documents = [point for point in results.points]

        # ReRank the documents
        reranked_docs = self.llm_manager.rerank_docs(documents, query)

        logger.info(f"Results generated: {len(reranked_docs)} documents retirved")
        return reranked_docs

    def sparse_search(self, query: str, pdf_id: str, brain_id: str):
        """
        Perform a sparse search based on the provided query using QdrantVectorStore.
        """
        # Generate sparse query
        sparse_query = list(settings.SPARSE_EMBEDDING_MODEL.embed([query]))[0]
        sparse_query = models.SparseVector(
            indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()
        )

        logger.info("Begin sparse Search")

        results = self.client.query_points(
            collection_name=brain_id,
            query=sparse_query,
            using="sparse",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.pdf_id", match=models.MatchValue(value=pdf_id)
                    )
                ]
            ),
        )
        documents = [point for point in results.points]

        logger.info(f"Sparse Search Completed. Result: {len(documents)} documents retrieved")
        return documents

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        # Format the prompt using the provided template
        formatted_prompt = self.prompt_template.format(
            question=question, context=context
        )

        # Call the invoke method with the formatted prompt string
        response = self.llm_manager.llm.invoke(formatted_prompt)
        logger.info("response generated")
        return response
