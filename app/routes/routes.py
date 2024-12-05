from fastapi import APIRouter
from controllers.pdf_controller import PdfController
from qdrant_client import QdrantClient 

# Create an APIRouter to register the routes
router = APIRouter()

# Create Qdrant client instance
client = QdrantClient(url="http://qdrant:6333")

# Create an instance of PdfController with injected dependencies
pdf_controller = PdfController(client)

# Register the routes with the router
router.post("/api/create-brain")(pdf_controller.create_new_brain)
router.get("/api/list-brains")(pdf_controller.list_brains)
router.post("/api/{brain_id}/upload")(pdf_controller.process_files)
router.get("/api/{brain_id}/list-files")(pdf_controller.list_files)
router.post("/api/{brain_id}/hybrid_rag")(pdf_controller.hybrid_rag_endpoint)
router.post("/api/{brain_id}/hyde_rag")(pdf_controller.hyde_rag_endpoint)
router.post("/api/{brain_id}/dense_rag")(pdf_controller.dense_rag_endpoint)
# router.post("/api/{brain_id}/multiquery_rag")(pdf_controller.multiquery_rag_endpoint)
router.post("/api/{brain_id}/all")(pdf_controller.all_endpoints)
router.post("/api/evaluate")(pdf_controller.evaluate_responses)