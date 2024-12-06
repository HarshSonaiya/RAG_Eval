from qdrant_client import QdrantClient, models
import logging
from typing import List, Set
import uuid 
import logging

from utils.helper import send_response, handle_exception

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

        if brain_name in existing_collections:
            logger.info(f"Brain with {brain_name} already exists.")
            return send_response(
                False, 
                200,
                f"Brain: {brain_name} already exists,",
                None
            )
            
        # If the brain does not exist, create a new collection
        try:
            brain_id = str(uuid.uuid4())  

            # Create a hybrid collection (dense + sparse)
            self.client.create_collection(
                collection_name=brain_id,
                vectors_config={
                    'dense': models.VectorParams(size=768, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                }
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
            logger.info(f"Created hybrid collection alias: {brain_name}")
            
            return send_response(
                True,
                200,
                f"Brain {brain_name} created in qdrant successfully.",
                {
                    "brain_id":brain_id
                }
            )  
        
        except Exception as e:
            logger.exception(f"Error while creating collections for brain '{brain_name}': {e}")
            return handle_exception(
                500,
                f"Error while creating collections for brain {brain_name}",
                e
            )

    async def list_brains(self):

        existing_collections =  self.client.get_collections().collections
        if not existing_collections:
                return []

        brain_info = []

        for collection in existing_collections :
            try:
                result = self.client.scroll(
                    collection_name = collection.name,
                    limit=1,
                    with_payload=True
                ) 
                points, _ = result 

                for point in points:
                    if "brain_name" in point.payload:
                        brain_info.append({
                            "brain_name": point.payload["brain_name"],
                            "brain_id": collection.name  
                        })
            except Exception as e:
                logger.info(f"Error searching in collection '{collection.name}': {e}")
                continue
        return brain_info

    async def list_files(self, brain_id: str):
        file_info = []
        file_names_set = set()
        
        try:
            result = self.client.scroll(
                collection_name = brain_id,
                with_payload=True,
                limit=4000
            ) 
            points, _ = result 
            for point in points:
                if "metadata" in point.payload and "file_name" in point.payload["metadata"] and "pdf_id" in point.payload["metadata"]:
                    
                    file_name = point.payload["metadata"]["file_name"]
                    
                    # Check if the file name is unique
                    if file_name not in file_names_set:
                        # If unique, add to the list and set
                        file_info.append({
                            "file_name": file_name,
                            "file_id": point.payload["metadata"]["pdf_id"]
                        })
                        file_names_set.add(file_name)
                    else:
                        logger.info(f"Skipping duplicate file: {file_name}")
                else:
                    logger.warning(f"Point {point.id} in collection '{brain_id}' missing 'file_name' or 'pdf_id' in metadata.")
            
            logger.info("File information", file_info)
            return file_info
        
        except Exception as e:
            logger.info(f"Error searching in collection '{brain_id}': {e}")
            return []

    async def check_files(self, file_name: str, brain_id: str):
        try:
            # Check if the file already exists in Qdrant
            existing_file_points = self.client.scroll(
                collection_name=brain_id,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.file_name", match=models.MatchValue(value=file_name))]
                ),
                limit=1  
            )
            logger.info("File check successfull.")
            return existing_file_points[0]
        except Exception as e :
            logger.error(f"Issue with existing file check: {str(e)}")
