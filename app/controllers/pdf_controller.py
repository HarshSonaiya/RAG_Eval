from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any

from app.services.pdf_service import PdfService
from app.services.qdrantclient import QdrantClientManager
from app.services.hybrid_rag_service import HybridRagService
from app.services.hyde_service import HyDEService
from app.services.dense_rag_service import DenseRagService
from app.services.multi_query_service import MultiQueryRetriever
from app.utils.llm_manager import LLMManager

# from app.services.evaluation import EvaluationService  


class PdfController:
    """Handles PDF processing and retrieval endpoints."""
    
    def __init__(self, evaluation_service: EvaluationService):
        
        self.pdf_service = PdfService()
        self.evaluation_service = evaluation_service

    async def process_files(self, files: List[UploadFile]) -> Any:
        """Helper function to process PDF files and index the content."""

        chunks = await self.pdf_service.process_pdf(files)
        return chunks

    async def handle_exception(self, e: Exception) -> Dict[str, Any]:
        """Handle exceptions and return an appropriate error response."""

        raise HTTPException(status_code=500, detail=str(e))

    async def send_for_evaluation(self, response: dict, query: str, model_type: str) -> Dict[str, Any]:
        """Send the response for evaluation and return the evaluation result."""
        evaluation_result = await self.evaluation_service.send_for_evaluation(response, query, model_type)
        return evaluation_result

    async def hybrid_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the hybrid RAG model."""
        try:
            chunks = self.process_files(files)
            await HybridRagService.index_hybrid_collection(chunks)
            combined_context = HybridRagService.hybrid_search(self.client, query)
            response = await HybridRagService.generate_response(query, combined_context)

            # Send the response for evaluation
            evaluation_result = await self.send_for_evaluation(response, query, "hybrid_rag")
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
            evaluation_result = await self.send_for_evaluation(response, query, "hyde_rag")
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
            evaluation_result = await self.send_for_evaluation(response, query, "hyde_rag")
            return {"response": response, "evaluation": evaluation_result}
        except Exception as e:
            return await self.handle_exception(e)

    async def multiquery_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            results = await self.pdf_service.process_pdfs_and_query(files, query)

            # Send the response for evaluation
            evaluation_result = await self.send_for_evaluation(results, query, "multiquery_rag")
            return {"response": results, "evaluation": evaluation_result}
        except Exception as e:
            return await self.handle_exception(e)


# Create an APIRouter to register the routes
router = APIRouter()

# Dependency injection for the LLM manager, Qdrant client, and evaluation service
llm_manager = LLMManager()
llm_chain_manager = LLMChainManager(llm_manager)
client_manager = QdrantClientManager()
evaluation_service = EvaluationService()

# Create an instance of PdfController with injected dependencies
pdf_controller = PdfController(llm_chain_manager, client_manager, evaluation_service)

# Register the routes with the router
router.post("/api/hybrid_rag")(pdf_controller.hybrid_rag_endpoint)
router.post("/api/hyde_rag")(pdf_controller.hyde_rag_endpoint)
router.post("/api/multiquery_rag")(pdf_controller.multiquery_rag_endpoint)
