import os
import shutil
from fastapi import UploadFile
from typing import List
from app.config.logging_config import LoggerFactory
from langchain.schema import Document  # Update with the correct import for your Document model
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PdfService:
    logger = LoggerFactory().get_logger("pdf_processing")

    @classmethod
    async def process_pdf(cls, files: List[UploadFile]) -> List[Document]:
        """
        Process uploaded PDF files, save them, and extract content chunks.

        Args:
            files (List[UploadFile]): List of uploaded PDF files.
            query (str): The user query (not used in current implementation).

        Returns:
            List[Document]: A list of Document chunks extracted from the PDFs.
        """
        all_chunks = []  # List to store chunks from all PDFs

        try:
            for file in files:
                temp_file_path = f"data/raw/{file.filename}"
                cls.logger.info(f"Received file: {file.filename}")

                # Ensure the directory exists
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                # Save the uploaded file to a temporary location
                with open(temp_file_path, "wb") as temp_file:
                    shutil.copyfileobj(file.file, temp_file)

                # Extract chunks from the saved PDF file
                chunks = cls.extract_content_from_pdf(temp_file_path)
                all_chunks.extend(chunks)

        except Exception as e:
            cls.logger.exception(f"An error occurred while processing the PDF: {e}")
            return {"error": str(e)}

        cls.logger.info("PDF processing completed successfully.")
        return all_chunks  # Return all extracted chunks

    @classmethod
    def extract_content_from_pdf(cls, file: str) -> List[Document]:
        """
        Extract and split content from a PDF file into chunks.

        Args:
            file (str): Path to the PDF file.

        Returns:
            List[Document]: A list of Documents containing various attributes
                            like page_content, metadata, etc. extracted from the PDF.
        """
        loader = PyPDFLoader(file)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        cls.logger.info(f"Chunking of {file} completed")
        return chunks
