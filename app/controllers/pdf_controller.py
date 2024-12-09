import json
import logging
import os
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
from fastapi import Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from services import (
    DenseRagService,
    HybridRagService,
    HyDEService,
    PdfService,
    evaluate_response,
)
from utils import Collection, LLMManager
from utils.helper import handle_exception, send_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline1.log"),
        logging.StreamHandler(),  # Log to the terminal (stdout)
    ],
)
logger = logging.getLogger("pipeline")


class PdfController:
    """Handles PDF processing and retrieval endpoints."""

    def __init__(self, client: QdrantClient, llm_manager: LLMManager):
        self.client = client
        self.pdf_service = PdfService(logger)
        self.collection = Collection(client)
        self.hybrid_rag_service = HybridRagService(client, llm_manager)
        self.hyde_service = HyDEService(client, llm_manager)
        self.dense_rag_service = DenseRagService(client, llm_manager)
        self.llm_manager = llm_manager

    async def create_new_brain(self, brain_name: str = Form(...)):
        """API to create a new brain."""
        try:
            brain_id = await self.collection.create_collections(brain_name)

            if brain_id:
                return send_response(
                    True,
                    201,
                    f"Brain {brain_name} created in qdrant successfully.",
                    brain_id,
                )
            else:
                return send_response(
                    False, 409, f"Brain: {brain_name} already exists", None
                )
        except Exception as e:
            logger.exception(f"Failed to create brain '{brain_name}': {e}")
            return handle_exception(
                500, f"Error while creating collections for brain {brain_name}, {e}"
            )

    async def list_brains(self):
        try:
            brain_info = await self.collection.list_brains()
            logger.info(f"Brain info: {brain_info}")

            if brain_info:
                return send_response(
                    True, 200, f"Brains fetched successfully.", brain_info
                )
            else:
                return send_response(False, 404, f"Create a brain first.", None)
        except Exception as e:
            return handle_exception(500, f"Error listing brains: {e}")

    async def list_files(self, brain_id: str):
        try:
            file_info = await self.collection.list_files(brain_id)

            if file_info:
                return send_response(
                    True, 200, f"Brains fetched successfully.", file_info
                )
            else:
                return send_response(
                    False, 404, f"Please upload some PDFs in the selected Brain.", None
                )
        except Exception as e:
            return handle_exception(500, f"Error listing files: {e}")

    async def process_files(self, files: List[UploadFile], brain_id: str) -> Any:
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

                existing_file_points = await self.collection.check_files(
                    file.filename, brain_id
                )

                if existing_file_points:
                    logger.info(
                        f"File {file.filename} already exists in the collection. Skipping."
                    )
                    continue

                # Generate a unique ID for the PDF
                pdf_id = str(uuid.uuid4())
                file_uuid_mapping[file.filename] = pdf_id
                logger.info(
                    "Generated unique ID for file %s: %s", file.filename, pdf_id
                )

                # Extract content chunks from the file
                chunks = await self.pdf_service.extract_content_from_pdf(file)

                # Assign metadata to chunks
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "pdf_id": pdf_id,
                            "file_name": file.filename,
                            "brain_id": brain_id,
                        }
                    )

                all_chunks.extend(chunks)
                logger.info(
                    "File %s processed and %d chunks generated.",
                    file.filename,
                    len(chunks),
                )

                # Update the data_registry collection
                await self.collection.update_registry(file.filename, pdf_id, brain_id)
            if all_chunks:
                success = await self.hybrid_rag_service.index_hybrid_collection(
                    all_chunks, brain_id
                )
                if success:
                    return send_response(
                        True, 201, f"Files Processed Successfully.", None
                    )
                else:
                    return send_response(
                        False, 422, f"PDF Content not supported for processing.", None
                    )
            else:
                logger.warning("No chunks to index.")
                return send_response(False, 404, f"No chunks to index.", None)

        except Exception as e:
            logger.exception("Error processing the PDF.")
            return handle_exception(500, f"{e} occured during PDF Processing.")

    async def hybrid_rag_endpoint(
        self, brain_id: str, payload: Dict[str, Any]
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
                        logger.info(
                            "Retrieved Context: %s", scored_point.payload["content"]
                        )
                        combined_context += scored_point.payload["content"] + " "
            if not combined_context:
                combined_context = ""

            logger.info("Begin Response generation in hybrid rag")
            response = self.hybrid_rag_service.generate_response(
                query, combined_context
            )
            response = response.content

            time.sleep(4)

            return send_response(
                True,
                200,
                f"Response generated successfully.",
                {
                    "hybrid_rag_response": response,
                    "hybrid_retriever_response": combined_context,
                },
            )
        except Exception as e:
            logger.exception("Error in hybrid RAG endpoint: %s", str(e))
            return handle_exception(500, f"Error generating hybrid response: {e}")

    async def hyde_rag_endpoint(
        self, brain_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handles requests for the HyDE RAG model."""
        try:
            query = payload.get("query")
            selected_pdfs = payload.get("selected_pdfs", [])

            # Ensure all selected PDFs are valid
            selected_pdf_ids = [pdf["file_id"] for pdf in selected_pdfs]

            hypothetical_document = self.hyde_service.generate_response(query, "")
            hypothetical_document = hypothetical_document.content
            logger.info("Hypothetical Document generated")

            dense_query = self.hybrid_rag_service.create_dense_vector(query)

            combined_context = ""
            for pdf_id in selected_pdf_ids:
                context = self.dense_rag_service.dense_search(
                    dense_query, pdf_id, brain_id
                )
                if context:
                    # ReRank the documents
                    reranked_docs = self.llm_manager.rerank_docs(context, query)
                    for scored_point in reranked_docs:
                        logger.info(
                            f"Retrieved Context: {scored_point.payload['content']}"
                        )
                        combined_context += scored_point.payload["content"] + " "
            if not combined_context:
                combined_context = ""
                
            logger.info(f"Combined Context {combined_context}")

            response = self.hyde_service.generate_response(query, combined_context)
            response = response.content
            logger.info("Response generated")

            return send_response(
                True,
                200,
                f"Response generated successfully.",
                {
                    "hyde_rag_response": response,
                    "hyde_retriever_response": combined_context,
                },
            )
        except Exception as e:
            return handle_exception(500, f"Error generating hyde response: {e}")

    async def dense_rag_endpoint(
        self, brain_id: str, payload: Dict[str, Any]
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
                context = self.dense_rag_service.dense_search(
                    dense_query, pdf_id, brain_id
                )
                if context:
                    # ReRank the documents
                    reranked_docs = self.llm_manager.rerank_docs(context, query)
                    for scored_point in reranked_docs:
                        logger.info(
                            f"Retrieved Context: {scored_point.payload['content']}"
                        )
                        combined_context += scored_point.payload["content"] + " "
            if not combined_context:
                combined_context = ""

            logger.info(f"Combined Context{combined_context}")

            response = self.dense_rag_service.generate_response(query, combined_context)
            response = response.content

            return send_response(
                True,
                200,
                f"Response generated successfully.",
                {
                    "dense_rag_response": response,
                    "dense_retriever_response": combined_context,
                },
            )
        except Exception as e:
            return handle_exception(500, f"Error generating dense response: {e}")

    async def sparse_rag_endpoint(
        self, brain_id: str, payload: Dict[str, Any]
    )-> Dict[str, Any]:
        """Handles requests for the Sparse RAG model."""
        try:
            query = payload.get("query")
            selected_pdfs = payload.get("selected_pdfs", [])

            # Ensure all selected PDFs are valid
            selected_pdf_ids = [pdf["file_id"] for pdf in selected_pdfs]

            logger.info("Begin Search in Sparse rag")

            combined_context = ""
            for pdf_id in selected_pdf_ids:
                context = self.hybrid_rag_service.sparse_search(query, pdf_id, brain_id)
                if context:
                    for scored_point in context:
                        logger.info(
                            f"Retrieved Context: {scored_point.payload['content']}"
                        )
                        combined_context += scored_point.payload["content"] + " "
            if not combined_context:
                combined_context = ""

            logger.info("Begin Response generation in Sparse rag")
            response = self.hybrid_rag_service.generate_response(
                query, combined_context
            )
            response = response.content

            time.sleep(4)

            return send_response(
                True,
                200,
                f"Response generated successfully.",
                {
                    "sparse_rag_response": response,
                    "sparse_retriever_response": combined_context,
                },
            )
        except Exception as e:
            logger.exception("Error in Sparse RAG endpoint: %s", str(e))
            return handle_exception(500, f"Error generating Sparse response: {e}")
   
    async def all_endpoints(
        self, brain_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handles requests for the Multiquery RAG model."""
        try:
            # Call each endpoint method and collect the responses
            hybrid_response = await self.hybrid_rag_endpoint(
                brain_id=brain_id, payload=payload
            )
            logger.info(
                "Hybrid RAG Implemented %s %s", hybrid_response, dir(hybrid_response)
            )

            hyde_response = await self.hyde_rag_endpoint(
                brain_id=brain_id, payload=payload
            )
            logger.info("HyDE RAG Implemented")

            dense_response = await self.dense_rag_endpoint(
                brain_id=brain_id, payload=payload
            )
            logger.info("Dense RAG Implemented")

            sparse_response = await self.sparse_rag_endpoint(
                brain_id=brain_id, payload=payload
            )
            logger.info("Sparse RAG Implemented")

            # Decode the byte responses to strings and parse them as JSON
            hybrid_rag_response = (
                json.loads(hybrid_response.body.decode("utf-8"))
                if hybrid_response.body
                else {}
            )
            hyde_rag_response = (
                json.loads(hyde_response.body.decode("utf-8"))
                if hyde_response.body
                else {}
            )
            dense_rag_response = (
                json.loads(dense_response.body.decode("utf-8"))
                if dense_response.body
                else {}
            )
            sparse_rag_response=(
                json.loads(dense_response.body.decode("utf-8"))
                if sparse_response.body
                else {}
            )

            # Combine the decoded responses into a single dictionary
            response_data = {
                "hybrid": hybrid_rag_response.get('data', {}),
                "hyde": hyde_rag_response.get('data', {}),
                "dense": dense_rag_response.get('data', {}),
                "sparse":sparse_rag_response.get('data',{})
            }
            # Combine the responses into a single dictionary
            return send_response(
                True,
                200,
                f"Response generated successfully.",
                response_data
            )
        except Exception as e:
            return handle_exception(500, f"Error generating response: {e}")

    async def send_for_evaluation(
        self, retrieved_context: str, query: str, llm_response: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Send the response for evaluation and return the evaluation result."""
        try:
            evaluation_result = await evaluate_response(
                retrieved_context, query, llm_response, ground_truth
            )
            evaluation_contents = [eval_msg.content for eval_msg in evaluation_result[0]], [
                eval_msg.content for eval_msg in evaluation_result[1]
            ]
            return send_response(True, 200, "Responses evaluated successfully.", evaluation_contents)
        except Exception as e:
            return handle_exception(500, f"Error evaluating response: {e}")

    async def evaluate_file(self, file: UploadFile):
        if not file.filename.endswith(".xlsx"):
            raise HTTPException(
                status_code=400, detail="Only .xlsx files are supported"
            )

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
            raise HTTPException(
                status_code=400, detail=f"Missing required sheets: {str(e)}"
            )

        brain_id = "ea8151ae-bd89-4eba-a9e8-d133e42a2e05"
        selected_pdfs = [
            {
                "file_name": "7181-attention-is-all-you-need.pdf",
                "file_id": "8c26fd2a-7724-40d2-97c9-678e67c1c2f5",
            },
            {
                "file_name": "IP Exam Preparation Framework.pdf",
                "file_id": "7795a29d-a480-422e-b7ed-797c4830a67a",
            },
            {"file_name": "cs.pdf", "file_id": "ce34380c-ba29-474e-875c-d021e8f09c7e"},
        ]

        # Evaluate LLM and Retriever Responses
        for index, row in llm_sheet.iterrows():

            question = row["Question"]
            ground_truth = row["Ground Truth"]

            payload = {
                "query": question,
                "selected_pdfs": selected_pdfs,
            }

            logger.info("Question: %s", question)
            logger.info("Ground truth: %s", ground_truth)

            results = await self.hybrid_rag_endpoint(brain_id=brain_id, payload=payload)
            llm_response = results.get("hybrid_rag_response", "No response available.")
            logger.info("LLM Response: %s", llm_response)
            retrieved_context = results.get(
                "hybrid_retriever_response", "No response available."
            )

            if pd.isna(question) or pd.isna(ground_truth) or pd.isna(llm_response):
                continue  # Skip rows with missing data

            # Call send_for_evaluation to evaluate both responses
            evaluation_result = await self.send_for_evaluation(
                retrieved_context, question, llm_response, ground_truth
            )

            # Extract LLM and Retriever evaluations
            llm_eval = evaluation_result[0]  # LLM evaluation results, list of strings
            eval_dict = {}
            for item in llm_eval[0].split(","):
                key, value = item.split(":")
                eval_dict[key] = float(value)

            # Update LLM sheet with response and evaluation metrics
            llm_sheet.at[index, "LLM Response"] = llm_response
            llm_sheet.at[index, "Helpfulness"] = float(
                eval_dict.get("helpfulness", 0.0)
            )
            llm_sheet.at[index, "Correctness"] = float(
                eval_dict.get("correctness", 0.0)
            )
            llm_sheet.at[index, "Coherence"] = float(eval_dict.get("coherence", 0.0))
            llm_sheet.at[index, "Complexity"] = float(eval_dict.get("complexity", 0.0))
            llm_sheet.at[index, "Verbosity"] = float(eval_dict.get("verbosity", 0.0))

            retriever_eval = evaluation_result[1]  # Retriever evaluation results,

            for item in retriever_eval[0].split(","):
                key, value = item.split(":")
                eval_dict[key] = float(value)

            # Update Retriever sheet with response and evaluation metrics
            retriever_sheet.at[index, "Retriever Response"] = retrieved_context
            retriever_sheet.at[index, "Helpfulness"] = float(
                eval_dict.get("helpfulness", 0.0)
            )
            retriever_sheet.at[index, "Correctness"] = float(
                eval_dict.get("correctness", 0.0)
            )
            retriever_sheet.at[index, "Coherence"] = float(
                eval_dict.get("coherence", 0.0)
            )
            retriever_sheet.at[index, "Complexity"] = float(
                eval_dict.get("complexity", 0.0)
            )
            retriever_sheet.at[index, "Verbosity"] = float(
                eval_dict.get("verbosity", 0.0)
            )

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
            return FileResponse(
                output_file,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="evaluated_test_set.xlsx",
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process the file.")
