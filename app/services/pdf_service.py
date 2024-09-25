import os
import shutil
from typing import List
from langchain.docstore.document import Document
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PdfService:
    def __init__(self, logger):
        self.logger = logger
        # Directory path inside the Docker volume (mapped to the host)
        self.docker_volume_path = "/data/raw"  # This is the path inside the Docker container
        self.logger.info("Initialized PdfService with Docker volume path: %s", self.docker_volume_path)

    def extract_content_from_pdf(self, file: str) -> List[Document]:
        """
        Extract and split content from a PDF file into chunks.

        Args:
            file (str): Path to the PDF file.

        Returns:
            List[Document]: A list of Documents containing various attributes
            like page_content, metadata, etc. extracted from the PDF.
        """
        self.logger.info("Extracting content from PDF file: %s", file)
        loader = PyPDFLoader(file)
        docs = loader.load()
        
        # Initialize the text splitter with specified chunk size and overlap
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)  # Split documents into chunks
        self.logger.info("Successfully extracted and split PDF into %d chunks.", len(chunks))
        
        return chunks
    
    async def process_pdf(self, files: List[UploadFile]) -> List[Document]:
        """
        Process uploaded PDF files, save them in a Docker volume, and extract content chunks.
        If a file is already stored, skip the extraction.

        Args:
            files (List[UploadFile]): List of uploaded PDF files.

        Returns:
            List[Document]: A list of Document chunks extracted from the PDFs.
        """
        all_chunks = []
        self.logger.info("Starting to process %d PDF files.", len(files))

        for file in files:
            try:
                # Path where the file will be saved inside the Docker container
                temp_file_path = os.path.join(self.docker_volume_path, file.filename)
                self.logger.info("Received file: %s", file.filename)

                # Ensure the directory exists inside the Docker volume
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                # Check if the file already exists to avoid redundant processing
                if not os.path.exists(temp_file_path):
                    # Save the uploaded file to the Docker volume (inside the container)
                    with open(temp_file_path, "wb") as temp_file:
                        shutil.copyfileobj(file.file, temp_file)
                    self.logger.info("Saved file %s to %s", file.filename, temp_file_path)

                    # Extract chunks from the saved PDF file
                    chunks = self.extract_content_from_pdf(temp_file_path)
                    all_chunks.extend(chunks)
                    self.logger.info("File %s processed and indexed with %d chunks.", file.filename, len(chunks))
                else:
                    self.logger.info("File %s already exists in volume, skipping processing.", file.filename)

            except Exception as e:
                self.logger.exception("Error processing the PDF %s: %s", file.filename, e)
                raise RuntimeError(f"Failed to process PDF {file.filename}")

        self.logger.info("PDF processing completed successfully with a total of %d chunks extracted.", len(all_chunks))
        return all_chunks
