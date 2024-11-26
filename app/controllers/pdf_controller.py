from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient 
from uuid import uuid4
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryByteStore

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

    async def list_file(self):
        file_info = await self.pdf_service.list_files()
        return file_info
    
    async def process_files(self, files: List[UploadFile] ) -> Any:
        """Helper function to process PDF files and index the content."""
        try:
                
            chunks, file_uuid_mapping = await self.pdf_service.process_pdf(files)
            
            if not chunks:
                self.logger.warning("No chunks were generated from the uploaded files.")
                return file_uuid_mapping
            # Generate and store embeddings in both collections
            hybrid_service = HybridRagService(client)  # Replace `client` with your Qdrant or embedding service instance
            dense_service = DenseRagService(client)

            await hybrid_service.index_hybrid_collection(chunks)
            await dense_service.index_dense_collection(chunks)
            
            self.logger.info(f"Embeddings generated and stored for file:")
            return file_uuid_mapping
        except Exception as e:
            self.logger.exception("Error processing the PDF %s: %s")
            raise RuntimeError(f"Failed to process PDF ")

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

    async def hybrid_rag_endpoint(self, query: str = Form(...), selected_pdf: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the hybrid RAG model."""
        try:    
            if selected_pdf not in self.file_uuid_mapping:
                raise ValueError(f"File {selected_pdf} not found in the system.")

            selected_pdf_id = self.file_uuid_mapping[selected_pdf]
            
            self.logger.info("Starting retrieval process for selected PDF ID: %s", selected_pdf)
            
            self.logger.info("Begin Indexing in hybrid rag")
            service = HybridRagService(client)
                
            self.logger.info("Begin Search in hybrid rag")
            combined_context = service.hybrid_search(query, selected_pdf_id)
            
            if not combined_context:
                self.logger.warning("No context found for query: %s and PDF ID: %s", query, selected_pdf_id)
                return {"error": "No context found for the given query and PDF."}

            self.logger.info("Begin Response generation in hybrid rag")
            response = service.generate_response(query, combined_context)
            response = response.content
            
            # Send the response for evaluation
            self.logger.info("Begin evaluation in hybrid rag")
            evaluation_results = await self.send_for_hybrid_evaluation(combined_context, query, response)

            return {
            "hybrid_rag_response": response, 
            "hybrid_retriever_response":combined_context,
            "hybrid_rag_llm_eval": evaluation_results[0],
            "hybrid_rag_retriever_eval": evaluation_results[1]
            }            
        except Exception as e:
            self.logger.exception("Error in hybrid RAG endpoint: %s", str(e))
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
            "hyde_retriever_response":combined_context,
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
            "dense_retriever_response":combined_context,
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
            combined_context = multiquery.retriever.invoke(query)
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
        
    async def multivector_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            multiquery = MultiQueryService(client)
            self.logger.info("DOCS RECEIVED")
            docs = await self.process_files(files, "multivector")
            self.logger.info(f"{len(docs)}")

            summary_query = "Generate a summary for the provided text" 
            summaries = []
            for doc in docs:
                summaries.append(multiquery.generate_response(summary_query, doc.page_content))

            doc_ids = [str(uuid4()) for _ in docs]

            store = InMemoryByteStore()
            id_key = "doc_id"

            # The retriever (empty to start)
            retriever = MultiVectorRetriever(
                vectorstore=multiquery.vector_store,
                byte_store=store,
                id_key=id_key,
            )

            doc_ids = [str(uuid4()) for _ in docs]
            summary_docs = [
                Document(page_content=s, metadata={"doc_id": doc_ids[i]})
                for i, s in enumerate(summaries)
            ]

            retriever.vectorstore.add_documents(summary_docs)

            sub_docs = retriever.vectorstore.similarity_search("LSTMS")
            print(f"Multivector Retriever Response: {sub_docs}")
            
            retrieved_docs = retriever.invoke("justice breyer")
            print(f"Multivector Response: {retrieved_docs}")

            # return {
            # "multiquery_rag_response": response, 
            # "multiquery_rag_llm_eval": evaluation_results[0],
            # "multiquery_rag_retriever_eval": evaluation_results[1]
            # }
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
router.post("/api/upload")(pdf_controller.process_files)
router.get("/api/list-files")(pdf_controller.list_file)
router.post("/api/hybrid_rag")(pdf_controller.hybrid_rag_endpoint)
router.post("/api/hyde_rag")(pdf_controller.hyde_rag_endpoint)
router.post("/api/dense_rag")(pdf_controller.dense_rag_endpoint)
router.post("/api/multiquery_rag")(pdf_controller.multiquery_rag_endpoint)
router.post("/api/all")(pdf_controller.all_endpoints)