from fastapi import  File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from qdrant_client import QdrantClient 
from langchain.schema import Document
import logging
import uuid
import os 

from services import PdfService, HybridRagService, HyDEService, DenseRagService, evaluate_response
from utils import LLMManager, Collection

import pandas as pd
from io import BytesIO

import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("logs/pipeline1.log"),
        logging.StreamHandler()  # Log to the terminal (stdout)
    ]
)
logger = logging.getLogger("pipeline")

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
                        logger.info("Retrieved Context: %s", scored_point.payload['content'])
                        combined_context += scored_point.payload['content'] + " "
            if not combined_context:
                return {"error": "No context found for the given query and PDF."}

            logger.info("Begin Response generation in hybrid rag")
            response = self.hybrid_rag_service.generate_response(query, combined_context)
            response = response.content

            time.sleep(4)
            
            return {
            "hybrid_rag_response": response, 
            "hybrid_retriever_response":combined_context,
            }            
        except Exception as e:
            logger.exception("Error in hybrid RAG endpoint: %s", str(e))
            # return await self.handle_exception(e)

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

        brain_id = "ea8151ae-bd89-4eba-a9e8-d133e42a2e05"
        selected_pdfs =[
            {
                "file_name":"7181-attention-is-all-you-need.pdf", 
                "file_id":"8c26fd2a-7724-40d2-97c9-678e67c1c2f5"
            },
            {
                "file_name":"IP Exam Preparation Framework.pdf",
                "file_id":"7795a29d-a480-422e-b7ed-797c4830a67a"
            },
            {
                "file_name":"cs.pdf",
                "file_id":"ce34380c-ba29-474e-875c-d021e8f09c7e"
            }
        ]

        # Evaluate LLM and Retriever Responses
        for index, row in llm_sheet.iterrows() :
            
            question = row["Question"]
            ground_truth = row["Ground Truth"]

            payload = {
                "query": question,
                "selected_pdfs": selected_pdfs,  
            }

            logger.info("Question: %s", question)
            logger.info("Ground truth: %s", ground_truth)

            results = await self.hybrid_rag_endpoint(brain_id=brain_id,payload=payload)
            llm_response = results.get("hybrid_rag_response", "No response available.")
            logger.info("LLM Response: %s", llm_response)
            retrieved_context = results.get("hybrid_retriever_response", "No response available.")
                
            if pd.isna(question) or pd.isna(ground_truth) or pd.isna(llm_response):
                continue  # Skip rows with missing data

            # Call send_for_evaluation to evaluate both responses
            evaluation_result = await self.send_for_evaluation(retrieved_context, question, llm_response, ground_truth)
            
            # Extract LLM and Retriever evaluations
            llm_eval = evaluation_result[0]  # LLM evaluation results, list of strings 
            eval_dict = {}
            for item in llm_eval[0].split(','):
                key, value = item.split(':')
                eval_dict[key] = float(value)

            # Update LLM sheet with response and evaluation metrics
            llm_sheet.at[index, "LLM Response"] = llm_response
            llm_sheet.at[index, "Helpfulness"] = float(eval_dict.get("helpfulness", 0.0))
            llm_sheet.at[index, "Correctness"] = float(eval_dict.get("correctness", 0.0))
            llm_sheet.at[index, "Coherence"] = float(eval_dict.get("coherence", 0.0))
            llm_sheet.at[index, "Complexity"] = float(eval_dict.get("complexity", 0.0))
            llm_sheet.at[index, "Verbosity"] = float(eval_dict.get("verbosity", 0.0))

            retriever_eval = evaluation_result[1]  # Retriever evaluation results, 

            for item in retriever_eval[0].split(','):
                key, value = item.split(':')
                eval_dict[key] = float(value)

            # Update Retriever sheet with response and evaluation metrics
            retriever_sheet.at[index, "Retriever Response"] = retrieved_context
            retriever_sheet.at[index, "Helpfulness"] = float(eval_dict.get("helpfulness", 0.0))
            retriever_sheet.at[index, "Correctness"] = float(eval_dict.get("correctness", 0.0))
            retriever_sheet.at[index, "Coherence"] = float(eval_dict.get("coherence", 0.0))
            retriever_sheet.at[index, "Complexity"] = float(eval_dict.get("complexity", 0.0))
            retriever_sheet.at[index, "Verbosity"] = float(eval_dict.get("verbosity", 0.0))

        # Save the updated data to a new Excel file
        current_directory = os.getcwd()

        # Define the output file path in the current working directory
        output_file = os.path.join(current_directory, "evaluated_test_set.xlsx")
        
        # Ensure the current directory exists (optional check)
        if not os.path.exists(current_directory):
            os.makedirs(current_directory)
            
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            llm_sheet.to_excel(writer, index=False, sheet_name="LLM Eval")
            retriever_sheet.to_excel(writer, index=False, sheet_name="Retriever Eval")
            
        if os.path.exists("evaluated_test_set.xlsx"):
            return FileResponse(output_file, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="evaluated_test_set.xlsx")
        else:
            raise HTTPException(status_code=500, detail="Failed to process the file.")
        