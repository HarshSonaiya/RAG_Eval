# services/hybrid_search_service.py
from app.config.settings import HYBRID_COLLECTION, DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL
from qdrant_client import QdrantClient, models

class HybridSearchService:

    @staticmethod
    def hybrid_search(client: QdrantClient, query: str, limit=5):
        dense_query = list(DENSE_EMBEDDING_MODEL.encode(query))
        sparse_query = list(SPARSE_EMBEDDING_MODEL.embed([query]))[0]

        sparse_query = models.SparseVector(
            indices=sparse_query.indices.tolist(),
            values=sparse_query.values.tolist()
        )

        results = client.query_points(
            collection_name=HYBRID_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=limit
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=limit
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF)
        )

        documents = [point for point in results.points]
        combined_context = "\n".join([doc.payload.get("content", "") for doc in results["combined_results"]])

        return combined_context
