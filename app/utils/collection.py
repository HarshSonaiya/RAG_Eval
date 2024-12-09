import logging
import uuid
from typing import List, Set

from config.settings import settings
from qdrant_client import QdrantClient, models

# Initialize logger
logger = logging.getLogger("pipeline")


class Collection:
    def __init__(self, client: QdrantClient):
        self.client = client

    async def create_collections(self, brain_name: str):
        """
        Create a dense and hybrid collection in Qdrant if it does not exist.
        """
        # Get the list of existing collections
        existing_collections = self.client.get_aliases()
        logger.info(
            "List of collections with aliases:%s  %s",
            existing_collections,
            type(existing_collections),
        )

        for alias in existing_collections.aliases:
            if alias.alias_name == brain_name:
                logger.info(f"Brain with {brain_name} already exists.")
                return {}

        # If the brain does not exist, create a new collection
        try:
            brain_id = str(uuid.uuid4())

            # Create a hybrid collection (dense + sparse)
            self.client.create_collection(
                collection_name=brain_id,
                vectors_config={
                    "dense": models.VectorParams(
                        size=768, distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
            logger.info(f"Created hybrid collection with ID: {brain_id}")

            # Create collection alias for collection identification
            self.client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=brain_id, alias_name=brain_name
                        )
                    )
                ]
            )
            logger.info(f"Created hybrid collection with alias: {brain_name}")

            return brain_id

        except Exception as e:
            logger.exception(
                f"Error while creating collections for brain '{brain_name}': {e}"
            )
            raise e

    async def list_brains(self):

        existing_collections = self.client.get_aliases().aliases
        if not existing_collections:
            return []

        brain_info = []

        for collection in existing_collections:
            brain_info.append(
                {
                    "brain_name": collection.alias_name,
                    "brain_id": collection.collection_name,
                }
            )

        return brain_info

    async def update_registry(self, file_name: str, pdf_id: str, brain_id: str) -> None:
        try:
            point_id = str(uuid.uuid4())
            payload = {
                "file_name": file_name,
                "pdf_id": pdf_id,
                "brain_id": brain_id,
            }

            # Upsert the record into the "data_registry" collection
            self.client.upsert(
                collection_name=settings.QDRANT_RECORD_STORE,
                points=[models.PointStruct(id=point_id, vector={}, payload=payload)],
            )

            logger.info(
                f"Successfully updated 'data_registry' with file '{file_name}' for brain ID '{brain_id}'."
            )
        except Exception as e:
            logger.error(
                f"Error updating 'data_registry' for file '{file_name}': {str(e)}"
            )
            raise e

    async def list_files(self, brain_id: str) -> list:
        """
        List all files in the DATA_REGISTRY collection for the specified brain_id.

        Args:
            brain_id (str): Brain ID for the collection.

        Returns:
            list: List of file information (file_name and file_id).
        """
        try:
            # Retrieve the total number of points in the collection
            point_count = self.client.count(
                collection_name=settings.QDRANT_RECORD_STORE,
            ).count

            if point_count == 0:
                point_count += 1

            # Retrieve all points with a filter for the given brain_id
            response, _ = self.client.scroll(
                collection_name=settings.QDRANT_RECORD_STORE,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="brain_id", match=models.MatchValue(value=brain_id)
                        )
                    ]
                ),
                limit=point_count,
            )

            # Extract file information
            file_info = [
                {
                    "file_name": point.payload.get("file_name"),
                    "file_id": point.payload.get("pdf_id"),
                }
                for point in response
                if "file_name" in point.payload and "pdf_id" in point.payload
            ]

            logger.info(f"Retrieved {len(file_info)} files for brain '{brain_id}'.")
            return file_info
        except Exception as e:
            logger.error(f"Error listing files for brain '{brain_id}': {str(e)}")
            return []

    async def check_files(self, file_name: str, brain_id: str):
        try:
            point_count = self.client.count(
                collection_name=settings.QDRANT_RECORD_STORE,
            ).count

            if point_count == 0:
                point_count += 1

            # Check if the file already exists in Qdrant
            existing_file_points, _ = self.client.scroll(
                collection_name=settings.QDRANT_RECORD_STORE,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name", match=models.MatchValue(value=file_name)
                        ),
                        models.FieldCondition(
                            key="brain_id", match=models.MatchValue(value=brain_id)
                        ),
                    ]
                ),
                limit=point_count,
            )

            file_exists = len(existing_file_points) > 0
            logger.info(
                f"File existence check for '{file_name}' in brain '{brain_id}' successful: {file_exists}"
            )
            return file_exists
        except Exception as e:
            logger.error(
                f"Error checking file '{file_name}' in brain '{brain_id}': {str(e)}"
            )
            raise e
