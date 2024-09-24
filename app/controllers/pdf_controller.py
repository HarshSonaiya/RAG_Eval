from fastapi import APIRouter, File, UploadFile, Form
from typing import List

from app.services.pdf_service import PdfService
from app.services.qdrantclient import QdrantClientManager
from rag.Hybrid_rag.index import Indexer
from rag.Hybrid_rag.retriever import HybridSearchService
from rag.Hybrid_rag.generator import LLMManager, LLMChainManager

router = APIRouter()

# Get or create a singleton Qdrant client
client = QdrantClientManager.get_client()

# Initialize LLMManager and LLMChainManager
llm_manager = LLMManager()
llm_chain_manager = LLMChainManager(llm_manager)

class PdfController:
    @classmethod
    @router.post(f"/api/hybrid_rag")
    async def hybrid_rag_endpoint(cls, files: List[UploadFile] = File(...), query: str = Form(...)):
        try:
            # Process the PDF and query
            chunks = await PdfService.process_pdf(files)

            # Index the hybrid collection with the Qdrant client
            Indexer.index_hybrid_collection(client, chunks)

            # Perform a hybrid search to get the combined context
            combined_context = HybridSearchService.hybrid_search(client, query)

            # Generate a response using the LLM chain manager
            response, combined_context = llm_chain_manager.answer_query(query, combined_context)

            return response
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    @router.post(f"/api/hyde_rag")
    async def hyde_rag_endpoint(cls, files: List[UploadFile] = File(...), query: str = Form(...)):
        try:
            # Process the PDF and query
            chunks = await PdfService.process_pdf(files)

            # Index the dense collection with the Qdrant client
            Indexer.index_hybrid_collection(client, chunks)

            # Generate Response using LLM chain Manager
            response, combined_context = llm_chain_manager.answer_hyde_query(query)


        except Exception as e:
            return {"error": str(e)}

    @classmethod
    @router.post(f"/api/multiquery_rag")
    async def multiquery_rag_endpoint(cls, files: List[UploadFile] = File(...), query: str = Form(...)):
        try:
            # Call the service layer to process the PDFs and query
            results = await PdfService.process_pdfs_and_query(files, query)
            return results
        except Exception as e:
            return {"error": str(e)}
