from controllers.pdf_controller import PdfController
from utils.llm_manager import LLMManager
from fastapi import APIRouter
from qdrant_client import QdrantClient

# Create an APIRouter to register the routes
router = APIRouter()

# Create Qdrant client instance
client = QdrantClient(url="http://qdrant:6333")

# Create an instance of LLMManager with injected dependencies
llm_manager = LLMManager()

# Create an instance of PdfController with injected dependencies
pdf_controller = PdfController(client, llm_manager)

# Register the routes with the router
router.post("/api/create-brain")(pdf_controller.create_new_brain)
router.get("/api/list-brains")(pdf_controller.list_brains)
router.post("/api/{brain_id}/upload")(pdf_controller.process_files)
router.get("/api/{brain_id}/list-files")(pdf_controller.list_files)
router.post("/api/{brain_id}/hybrid")(pdf_controller.hybrid_rag_endpoint)
router.post("/api/{brain_id}/sparse")(pdf_controller.sparse_rag_endpoint)
router.post("/api/{brain_id}/hyde")(pdf_controller.hyde_rag_endpoint)
router.post("/api/{brain_id}/dense")(pdf_controller.dense_rag_endpoint)
router.post("/api/{brain_id}/all")(pdf_controller.all_endpoints)
router.post("/api/evaluate_response")(pdf_controller.send_for_evaluation)
router.post("/api/evaluate-file")(pdf_controller.evaluate_file)
