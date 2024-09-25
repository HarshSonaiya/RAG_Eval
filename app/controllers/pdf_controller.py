from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any
from langchain.schema import Document

from services.pdf_service import PdfService
from services.qdrantclient import QdrantClientManager
from services.hybrid_rag_service import HybridRagService
from services.hyde_service import HyDEService
from services.dense_rag_service import DenseRagService
from services.multi_query_service import MultiQueryRetriever
from services.evaluation_service import evaluate_response
from config.logging_config import LoggerFactory  


class PdfController:
    """Handles PDF processing and retrieval endpoints."""
    
    def __init__(self):
        logger_factory = LoggerFactory()  
        logger = logger_factory.get_logger("pdf_processing")  

        self.pdf_service = PdfService(logger)  



    async def process_files(self, files: List[UploadFile]) -> Any:
        """Helper function to process PDF files and index the content."""

        chunks = await self.pdf_service.process_pdf(files)
        return chunks

    async def handle_exception(self, e: Exception) -> Dict[str, Any]:
        """Handle exceptions and return an appropriate error response."""

        raise HTTPException(status_code=500, detail=str(e))

    async def send_for_evaluation(self, retrieved_context: str, query: str, response: str) -> Dict[str, Any]:
        """Send the response for evaluation and return the evaluation result."""
        evaluation_result = await evaluate_response(retrieved_context, query, response)
        return evaluation_result

    async def hybrid_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the hybrid RAG model."""
        try:
            chunks = self.process_files(files)
            await HybridRagService.index_hybrid_collection(chunks)
            combined_context = HybridRagService.hybrid_search(self.client, query)
            response = await HybridRagService.generate_response(query, combined_context)

            # Send the response for evaluation
            evaluation_result = await self.send_for_evaluation(combined_context, query, response)
            return {"response": response, "evaluation": evaluation_result}
        except Exception as e:
            return await self.handle_exception(e)

    async def hyde_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            chunks = self.process_files(files)
            await DenseRagService.index_collection(chunks)
            hypothetical_document = await HyDEService.generate_response(query, "")
            combined_context = HyDEService.hyde_search(self.client, hypothetical_document)
            response = await HyDEService.generate_response(query, combined_context)

            # Send the response for evaluation
            evaluation_result = await self.send_for_evaluation(combined_context, query, response)
            return {"response": response, "evaluation": evaluation_result}
        except Exception as e:
            return await self.handle_exception(e)
    
    async def dense_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            chunks = self.process_files(files)
            await DenseRagService.index_collection(chunks)
            combined_context = HyDEService.hyde_search(self.client, query)
            response = await HyDEService.generate_response(query, combined_context)

            # Send the response for evaluation
            evaluation_result = await self.send_for_evaluation(combined_context, query, response)
            return {"response": response, "evaluation": evaluation_result}
        except Exception as e:
            return await self.handle_exception(e)

    async def multiquery_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            chunks = self.process_files(files)
            await DenseRagService.index_collection(chunks)
            combined_context = MultiQueryRetriever.hyde_search(self.client, query)
            response = await HyDEService.generate_response(query, combined_context)

            # Send the response for evaluation
            evaluation_result = await self.send_for_evaluation(combined_context, query, response)
            return {"response": response, "evaluation": evaluation_result}
        except Exception as e:
            return await self.handle_exception(e)
        
    async def all_endpoints(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            # Process the files
            chunks = await self.process_files(files)
            
            # Index the chunks (you may want to ensure this is done only once)
            await DenseRagService.index_collection(chunks)

            # Call each endpoint method and collect the responses
            hybrid_response = await self.hybrid_rag_endpoint(files, query)
            hyde_response = await self.hyde_rag_endpoint(files, query)
            dense_response = await self.dense_rag_endpoint(files, query)
            multiquery_response = await self.multiquery_rag_endpoint(files, query)

            # Combine the responses into a single dictionary
            return {
                "hybrid": hybrid_response,
                "hyde": hyde_response,
                "dense": dense_response,
                "multiquery": multiquery_response,
            }
        except Exception as e:
            return await self.handle_exception(e)


# Create an APIRouter to register the routes
router = APIRouter()

# Create an instance of PdfController with injected dependencies
pdf_controller = PdfController()

# Register the routes with the router
router.post("/api/hybrid_rag")(pdf_controller.hybrid_rag_endpoint)
router.post("/api/hyde_rag")(pdf_controller.hyde_rag_endpoint)
router.post("/api/dense_rag")(pdf_controller.dense_rag_endpoint)
router.post("/api/multiquery_rag")(pdf_controller.multiquery_rag_endpoint)
router.post("api/all")(pdf_controller.all_endpoints)