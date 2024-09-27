from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any
from qdrant_client import QdrantClient 

from services.pdf_service import PdfService
from services.hybrid_rag_service import HybridRagService
from services.hyde_service import HyDEService
from services.dense_rag_service import DenseRagService
from services.multi_query_service import MultiQueryService
from services.evaluation_service import  evaluate_hybrid_response, evaluate_response
from config.logging_config import LoggerFactory  


class PdfController:
    """Handles PDF processing and retrieval endpoints."""
    
    def __init__(self):
        logger_factory = LoggerFactory()  
        self.logger = logger_factory.get_logger("pdf_processing")  

        self.pdf_service = PdfService(self.logger)  

    async def process_files(self, files: List[UploadFile]) -> Any:
        """Helper function to process PDF files and index the content."""

        chunks = await self.pdf_service.process_pdf(files)
        return chunks

    async def handle_exception(self, e: Exception) -> Dict[str, Any]:
        """Handle exceptions and return an appropriate error response."""

        raise HTTPException(status_code=500, detail=str(e))
    
    async def send_for_hybrid_evaluation(self, retrieved_context: str, query: str, response: str) -> Dict[str, Any]:
        """Send the response for evaluation and return the evaluation result."""
        evaluation_result = await evaluate_hybrid_response(retrieved_context, query, response)
        evaluation_contents = [
            eval_msg.content for eval_msg in evaluation_result[0]
        ], [
            eval_msg.content for eval_msg in evaluation_result[1]
        ]
        return evaluation_contents
    
    async def send_for_evaluation(self, retrieved_context: str, query: str, response: str) -> Dict[str, Any]:
        """Send the response for evaluation and return the evaluation result."""
        evaluation_result = await evaluate_response(retrieved_context, query, response)
        evaluation_contents = [
            eval_msg.content for eval_msg in evaluation_result[0]
        ], [
            eval_msg.content for eval_msg in evaluation_result[1]
        ]
        return evaluation_contents

    async def hybrid_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the hybrid RAG model."""
        try:
            self.logger.info("Begin chunking in hybrid rag")
            chunks = await self.process_files(files)
 
            self.logger.info("Begin Indexing in hybrid rag")
            service = HybridRagService(client)  
            if chunks :
                await service.index_hybrid_collection(chunks)
                
            self.logger.info("Begin Search in hybrid rag")
            combined_context = service.hybrid_search(query)
            
            self.logger.info("Begin Response generation in hybrid rag")
            response = service.generate_response(query, combined_context)
            response = response.content
            
            self.logger.info("Begin evaluation in hybrid rag")
            # Send the response for evaluation
            evaluation_results = await self.send_for_hybrid_evaluation(combined_context, query, response)

            return {
            "hybrid_rag_response": response, 
            "hybrid_rag_llm_eval": evaluation_results[0],
            "hybrid_rag_retriever_eval": evaluation_results[1]
            }        
        except Exception as e:
            return await self.handle_exception(e)

    async def hyde_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            chunks = await self.process_files(files)
            dense_Service = DenseRagService(client)
            if chunks :
                await dense_Service.index_collection(chunks)
            hyde_Service = HyDEService(client)
            hypothetical_document = hyde_Service.generate_response(query, "")
            self.logger.info("Hypothetical Document generated")
            hypothetical_document = hypothetical_document.content
            combined_context = hyde_Service.hyde_search(hypothetical_document)
            combined_context = " ".join([doc.page_content for doc in combined_context])

            self.logger.info("Combined Context generated")

            response = hyde_Service.generate_response(query, combined_context)
            self.logger.info("Response generated")
            
            response = response.content 
            
            # Send the response for evaluation
            evaluation_results = await self.send_for_evaluation(combined_context, query, response)
            return {
            "hyde_rag_response": response, 
            "hyde_rag_llm_eval": evaluation_results[0],
            "hyde_rag_retriever_eval": evaluation_results[1]
            }        
        except Exception as e:
            return await self.handle_exception(e)
    
    async def dense_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            chunks = await self.process_files(files)
            dense_Service = DenseRagService(client)
            if chunks :
                await dense_Service.index_collection(chunks)
            combined_context = dense_Service.dense_search(query)
            combined_context = " ".join([doc.page_content for doc in combined_context])

            response = dense_Service.generate_response(query, combined_context)
            response = response.content 
            
            # Send the response for evaluation
            evaluation_results = await self.send_for_evaluation(combined_context, query, response)
            return {
            "dense_rag_response": response, 
            "dense_rag_llm_eval": evaluation_results[0],
            "dense_rag_retriever_eval": evaluation_results[1]
            }        
        except Exception as e:
            return await self.handle_exception(e)

    async def multiquery_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            chunks = await self.process_files(files)
            dense_Service = DenseRagService(client)
            if chunks :
                await dense_Service.index_collection(chunks)
            multiquery = MultiQueryService(client)
            combined_context = multiquery.multi_query_retriever.invoke(query)
            combined_context = " ".join([doc.page_content for doc in combined_context])

            self.logger.info("Retriever Task Successfull")
            response = multiquery.generate_response(query, combined_context)
            response = response.content 

            # # Send the response for evaluation
            evaluation_results = await self.send_for_evaluation(combined_context, query, response)
            return {
            "multiquery_rag_response": response, 
            "multiquery_rag_llm_eval": evaluation_results[0],
            "multiquery_rag_retriever_eval": evaluation_results[1]
            }
        except Exception as e:
            return await self.handle_exception(e)
        
    async def all_endpoints(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            # Process the files
            chunks = await self.process_files(files)
            dense_Service = DenseRagService(client)
            service = HybridRagService(client)  

            if chunks :
                await dense_Service.index_collection(chunks)
                await service.index_hybrid_collection(chunks)

            # Call each endpoint method and collect the responses
            hybrid_response = await self.hybrid_rag_endpoint(files, query)
            self.logger.info("Hybrid RAG Implemented")
            hyde_response = await self.hyde_rag_endpoint(files, query)
            self.logger.info("HyDE RAG Implemented")
            dense_response = await self.dense_rag_endpoint(files, query)
            self.logger.info("Dense RAG Implemented")
            multiquery_response = await self.multiquery_rag_endpoint(files, query)
            self.logger.info("ALL RAGS Implemented")

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
# Create Qdrant client instance
client = QdrantClient(url="http://qdrant:6333", port=6333)

# Register the routes with the router
router.post("/api/hybrid_rag")(pdf_controller.hybrid_rag_endpoint)
router.post("/api/hyde_rag")(pdf_controller.hyde_rag_endpoint)
router.post("/api/dense_rag")(pdf_controller.dense_rag_endpoint)
router.post("/api/multiquery_rag")(pdf_controller.multiquery_rag_endpoint)
router.post("/api/all")(pdf_controller.all_endpoints)