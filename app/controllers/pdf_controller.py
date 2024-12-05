from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any
from qdrant_client import QdrantClient 
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryByteStore
import logging
import uuid

from utils.llm_manager import LLMManager

from services.pdf_service import PdfService
from services.hybrid_rag_service import HybridRagService
from services.hyde_service import HyDEService
from services.dense_rag_service import DenseRagService
# from services.multi_query_service import MultiQueryService
from services.evaluation_service import evaluate_response
from utils.collection import Collection

import pandas as pd
from io import BytesIO


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PdfController:
    """Handles PDF processing and retrieval endpoints."""
    
    def __init__(self, client: QdrantClient):
        self.client = client
        self.pdf_service = PdfService(logger)
        self.collection = Collection(client)
        self.llm_manager = LLMManager()
        self.hybrid_rag_service = HybridRagService(client)
        self.hyde_service = HyDEService(client)
        self.dense_rag_service = DenseRagService(client)
        # self.multi_query_service = MultiQueryService(client)
    
    async def create_new_brain(self, brain_name: str = Form(...)):
        """API to create a new brain."""
        try:
            await self.collection.create_collections(brain_name)
            return {"status": "success"}
        except Exception as e:
            logger.exception(f"Failed to create brain '{brain_name}': {e}")
            raise HTTPException(status_code=500, detail="Error creating brain.")

    async def list_brains(self):
        try:
            brain_info = await self.collection.list_brains()
            logger.info(f"Brain info: {brain_info}")
            return brain_info
        except Exception as e:
            logger.exception("Error listing brains: %s", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve brains list.")

    async def list_files(self, brain_id: str):
        try:
            file_info = await self.collection.list_files(brain_id)
            return file_info
        except Exception as e:
            logger.exception("Error listing files: %s", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve file list.")
    
    async def process_files(self, files: List[UploadFile], brain_id: str ) -> Any:
        """
        Process PDF files, check for duplicates in Qdrant and index the content.
        
        Args:
            files (List[UploadFile]): List of uploaded PDF files.
            brain_id (str): The brain's unique identifier.

        Returns:
            dict: A dictionary containing the status and file-to-UUID mapping.
        """
        try:
            all_chunks = []
            file_uuid_mapping = {}

            logger.info("Starting to process PDF files for brain: %s", brain_id)

            for file in files:
                
                existing_file_points = await self.collection.check_files(file.filename, brain_id)
                
                if existing_file_points:  
                    logger.info(f"File {file.filename} already exists in the collection. Skipping.")
                    continue

                # Generate a unique ID for the PDF
                pdf_id = str(uuid.uuid4())
                file_uuid_mapping[file.filename] = pdf_id
                logger.info("Generated unique ID for file %s: %s", file.filename, pdf_id)

                # Extract content chunks from the file
                chunks = await self.pdf_service.extract_content_from_pdf(file)

                # Assign metadata to chunks
                for chunk in chunks:
                    chunk.metadata["pdf_id"] = pdf_id
                    chunk.metadata["file_name"] = file.filename
                    chunk.metadata["brain_id"] = brain_id

                all_chunks.extend(chunks)
                logger.info("File %s processed and indexed with %d chunks.", file.filename, len(chunks))
            
            if all_chunks: 
                await self.hybrid_rag_service.index_hybrid_collection(all_chunks, brain_id)
            else :
                logger.warning("No chunks to index.")

        except Exception as e:
            logger.exception("Error processing the PDF %s: %s")
            raise HTTPException(status_code=500, detail="Failed to process files.")

    async def handle_exception(self, e: Exception) -> Dict[str, Any]:
        """Handle exceptions and return an appropriate error response."""
        raise HTTPException(status_code=500, detail=str(e))
    
    async def send_for_evaluation(
        self, 
        retrieved_context: str, 
        query: str, 
        response: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Send the response for evaluation and return the evaluation result."""
        evaluation_result = await evaluate_response(retrieved_context, query, response, ground_truth)
        evaluation_contents = [
            eval_msg.content for eval_msg in evaluation_result[0]
        ], [
            eval_msg.content for eval_msg in evaluation_result[1]
        ]
        return evaluation_contents

    async def hybrid_rag_endpoint(
        self, 
        brain_id: str,        
        payload: Dict[str, Any] 
        ) -> Dict[str, Any]:
        """Handles requests for the hybrid RAG model."""
        try:    
            query = payload.get("query")
            selected_pdfs = payload.get("selected_pdfs", [])

            # Ensure all selected PDFs are valid
            selected_pdf_ids = [pdf["file_id"] for pdf in selected_pdfs]

            logger.info("Begin Search in hybrid rag")

            combined_context = ""
            for pdf_id in selected_pdf_ids:
                context = self.hybrid_rag_service.hybrid_search(query, pdf_id, brain_id)
                if context:
                    for scored_point in context:
                        logger.info("Retrieved Context:", scored_point.payload['content'])
                        combined_context += scored_point.payload['content'] + " "
            if not combined_context:
                return {"error": "No context found for the given query and PDF."}

            logger.info("Begin Response generation in hybrid rag")
            response = self.hybrid_rag_service.generate_response(query, combined_context)
            response = response.content
            
            # Send the response for evaluation
            logger.info("Begin evaluation in hybrid rag")
            evaluation_results = await self.send_for_evaluation(combined_context, query, response)

            return {
            "hybrid_rag_response": response, 
            "hybrid_retriever_response":combined_context,
            "hybrid_rag_llm_eval": evaluation_results[0],
            "hybrid_rag_retriever_eval": evaluation_results[1]
            }            
        except Exception as e:
            logger.exception("Error in hybrid RAG endpoint: %s", str(e))
            return await self.handle_exception(e)

    async def hyde_rag_endpoint(
        self,
        brain_id: str,        
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            query = payload.get("query")
            selected_pdfs = payload.get("selected_pdfs", [])

            # Ensure all selected PDFs are valid
            selected_pdf_ids = [pdf["file_id"] for pdf in selected_pdfs]

            hypothetical_document =self.hyde_service.generate_response(query, "")
            hypothetical_document = hypothetical_document.content
            logger.info("Hypothetical Document generated")

            dense_query = self.hybrid_rag_service.create_dense_vector(query)

            combined_context = ""
            for pdf_id in selected_pdf_ids:
                context = self.dense_rag_service.dense_search(dense_query, pdf_id, brain_id)
                if context:
                    # ReRank the documents 
                    reranked_docs = self.llm_manager.rerank_docs(context, query)
                    for scored_point in reranked_docs:
                        logger.info("Retrieved Context:", scored_point.payload['content'])
                        combined_context += scored_point.payload['content'] + " "
            if not combined_context:
                return {"error": "No context found for the given query and PDF."}
                                
            logger.info("Combined Context", combined_context)

            response = self.hyde_service.generate_response(query, combined_context)
            response = response.content 
            logger.info("Response generated")
            
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
    
    async def dense_rag_endpoint(
        self,
        brain_id: str,        
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            query = payload.get("query")
            selected_pdfs = payload.get("selected_pdfs", [])

            # Ensure all selected PDFs are valid
            selected_pdf_ids = [pdf["file_id"] for pdf in selected_pdfs]
            
            dense_query = self.hybrid_rag_service.create_dense_vector(query)
            
            combined_context = ""
            for pdf_id in selected_pdf_ids:
                context = self.dense_rag_service.dense_search(dense_query, pdf_id, brain_id)
                if context:
                    # ReRank the documents 
                    reranked_docs = self.llm_manager.rerank_docs(context, query)
                    for scored_point in reranked_docs:
                        logger.info("Retrieved Context:", scored_point.payload['content'])
                        combined_context += scored_point.payload['content'] + " "
            if not combined_context:
                return {"error": "No context found for the given query and PDF."}
                                
            logger.info("Combined Context", combined_context)

            response = self.dense_rag_service.generate_response(query, combined_context)
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

    # async def multiquery_rag_endpoint( 
    #     self,
    #     brain_id: str,        
    #     payload: Dict[str, Any]
    # ) -> Dict[str, Any]:
    #     """Handles requests for the Multiquery RAG model."""
    #     try:
    #         query = payload.get("query")
    #         selected_pdfs = payload.get("selected_pdfs", [])

    #          # Remove the "hybrid@" prefix if it exists
    #         if brain_id.startswith("hybrid@"):
    #             brain_id = brain_id[7:]

    #         # Ensure all selected PDFs are valid
    #         selected_pdf_ids = [pdf["file_id"] for pdf in selected_pdfs]
            
    #         combined_context = ""
    #         for pdf_id in selected_pdf_ids:
    #             context = self.multi_query_service.retriever.invoke(query)
    #         combined_context = " ".join([doc.page_content for doc in combined_context])

    #         logger.info("Retriever Task Successfull")
    #         response = self.multi_query_service.generate_response(query, combined_context)
    #         response = response.content 

    #         # # Send the response for evaluation
    #         evaluation_results = await self.send_for_evaluation(combined_context, query, response)
    #         return {
    #         "multiquery_rag_response": response, 
    #         "multiquery_rag_llm_eval": evaluation_results[0],
    #         "multiquery_rag_retriever_eval": evaluation_results[1]
    #         }
    #     except Exception as e:
    #         return await self.handle_exception(e)
        
    # # async def multivector_rag_endpoint(self, files: List[UploadFile] = File(...), query: str = Form(...)) -> Dict[str, Any]:
    #     """Handles requests for the Multiquery RAG model."""
    #     try:
    #         multiquery = MultiQueryService(client)
    #         logger.info("DOCS RECEIVED")
    #         docs = await self.process_files(files, "multivector")
    #         logger.info(f"{len(docs)}")

    #         summary_query = "Generate a summary for the provided text" 
    #         summaries = []
    #         for doc in docs:
    #             summaries.append(multiquery.generate_response(summary_query, doc.page_content))

    #         doc_ids = [str(uuid4()) for _ in docs]

    #         store = InMemoryByteStore()
    #         id_key = "doc_id"

    #         # The retriever (empty to start)
    #         retriever = MultiVectorRetriever(
    #             vectorstore=multiquery.vector_store,
    #             byte_store=store,
    #             id_key=id_key,
    #         )

    #         doc_ids = [str(uuid4()) for _ in docs]
    #         summary_docs = [
    #             Document(page_content=s, metadata={"doc_id": doc_ids[i]})
    #             for i, s in enumerate(summaries)
    #         ]

    #         retriever.vectorstore.add_documents(summary_docs)

    #         sub_docs = retriever.vectorstore.similarity_search("LSTMS")
    #         print(f"Multivector Retriever Response: {sub_docs}")
            
    #         retrieved_docs = retriever.invoke("justice breyer")
    #         print(f"Multivector Response: {retrieved_docs}")

    #         # return {
    #         # "multiquery_rag_response": response, 
    #         # "multiquery_rag_llm_eval": evaluation_results[0],
    #         # "multiquery_rag_retriever_eval": evaluation_results[1]
    #         # }
    #     except Exception as e:
    #         return await self.handle_exception(e)
            
    async def all_endpoints(
        self,
        brain_id: str,        
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:        
        """Handles requests for the Multiquery RAG model."""
        try:
            
            # Call each endpoint method and collect the responses
            hybrid_response = await self.hybrid_rag_endpoint(brain_id= brain_id, payload=payload)
            logger.info("Hybrid RAG Implemented")
            hyde_response = await self.hyde_rag_endpoint(brain_id= brain_id, payload=payload)
            logger.info("HyDE RAG Implemented")
            dense_response = await self.dense_rag_endpoint(brain_id= brain_id, payload=payload)
            logger.info("Dense RAG Implemented")
            # multiquery_response = await self.multiquery_rag_endpoint(query=query, selected_pdf=files[0].filename)
            # logger.info("ALL RAGS Implemented")

            # Combine the responses into a single dictionary
            return {
                "hybrid": hybrid_response,
                "hyde": hyde_response,
                "dense": dense_response,
                # "multiquery": multiquery_response,
            }
        except Exception as e:
            return await self.handle_exception(e)

    async def evaluate_responses(self, file: UploadFile):
        if not file.filename.endswith(".xlsx"):
            raise HTTPException(status_code=400, detail="Only .xlsx files are supported")

        # Read the uploaded Excel file
        content = await file.read()
        try:
            excel_data = pd.ExcelFile(BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Excel file: {str(e)}")

        # Parse the relevant sheets
        try:
            llm_sheet = excel_data.parse("LLM Eval")
            retriever_sheet = excel_data.parse("Retriever Eval")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Missing required sheets: {str(e)}")

        # Results lists for updating the sheets
        llm_results = []
        retriever_results = []

        # Evaluate LLM Responses
        for _, row in llm_sheet.iterrows():
            question = row["Question"]
            ground_truth = row["Ground Truth"]

            brain_id = "7760b47d-bd53-48af-be07-8b8054d8ff06"
            selected_pdfs =[
                {
                    "file_name":"7181-attention-is-all-you-need.pdf", 
                    "file_id":"d5c33a08-6396-42db-aeb2-ac6ed42088a6"
                },
                {
                    "file_name":"IP Exam Preparation Framework.pdf",
                    "file_id":"8c9e2fdd-5c1f-4bd1-a475-5351335d78dd"
                }
            ]

            payload = {
                "query": question,
                "selected_pdfs": selected_pdfs,  
            }
            results = await self.hybrid_rag_endpoint(brain_id=brain_id,payload=payload)
            llm_response = results.get("hybrid", {}).get("hybrid_rag_response", "No response available."),
            retrieved_context = results.get("hybrid", {}).get("hybrid_retriever_response", "No response available.")
                
            if pd.isna(question) or pd.isna(ground_truth) or pd.isna(llm_response):
                llm_results.append({"Evaluation Result": "Skipped - Missing Data"})
                continue  # Skip rows with missing data

            # Call send_for_evaluation
            evaluation_result = await self.send_for_evaluation(retrieved_context, question, llm_response, ground_truth)
            llm_results.append({"Evaluation Result": evaluation_result})

        # Add the results as a new column to the LLM sheet
        llm_sheet["Evaluation Result"] = [result["Evaluation Result"] for result in llm_results]

        # Evaluate Retriever Responses
        for _, row in retriever_sheet.iterrows():
            question = row["Question"]
            ground_truth = row["Ground Truth"]
            retriever_response = row["Retriever Response"] if "Retriever Response" in row else None
            retrieved_context = "Dummy Context for Retriever"  # Replace with actual context logic

            if pd.isna(question) or pd.isna(ground_truth) or pd.isna(retriever_response):
                retriever_results.append({"Evaluation Result": "Skipped - Missing Data"})
                continue  # Skip rows with missing data

            # Call send_for_evaluation
            evaluation_result = await send_for_evaluation(retrieved_context, question, retriever_response, ground_truth)
            retriever_results.append({"Evaluation Result": evaluation_result})

        # Add the results as a new column to the Retriever sheet
        retriever_sheet["Evaluation Result"] = [result["Evaluation Result"] for result in retriever_results]

        # Save the updated data to a new Excel file
        output_file = "/mnt/data/evaluated_test_set.xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            llm_sheet.to_excel(writer, index=False, sheet_name="LLM Eval")
            retriever_sheet.to_excel(writer, index=False, sheet_name="Retriever Eval")

        return {"message": "Evaluation completed", "file_path": output_file}



# Create an APIRouter to register the routes
router = APIRouter()

# Create Qdrant client instance
client = QdrantClient(url="http://qdrant:6333", port=6333)

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