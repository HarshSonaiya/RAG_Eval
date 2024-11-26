import os
import shutil
import json
import uuid
from typing import List, Optional
from langchain.docstore.document import Document
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.globals import set_debug

set_debug(True)

class PdfService:
    def __init__(self, logger):
        self.logger = logger
        # Directory path inside the Docker volume (mapped to the host)
        self.docker_volume_path = "/data/raw"  # This is the path inside the Docker container
        self.uuid_mapping_file = "/data/uuid_mapping.json"  # Path to store the UUID mapping
        self.file_uuid_mapping = self.load_uuid_mapping()  # Load UUID mapping from file on init
        self.logger.info("Initialized PdfService with Docker volume path: %s", self.docker_volume_path)

    def load_uuid_mapping(self):
        """Load the UUID mapping from a file."""
        if os.path.exists(self.uuid_mapping_file):
            with open(self.uuid_mapping_file, "r") as f:
                return json.load(f)
        return {}
    
    def save_uuid_mapping(self):
        """Save the UUID mapping to a file."""
        with open(self.uuid_mapping_file, "w") as f:
            json.dump(self.file_uuid_mapping, f)
    
    async def list_files(self):
        """List all the files in the server directory along with their UUID (pdf_id)."""
        try:
            files_info = []
            
            # Check if the PDF_DIRECTORY exists
            if not os.path.exists(self.docker_volume_path):
                raise FileNotFoundError(f"The directory {self.docker_volume_path} does not exist.")
            
            # List all files in the directory
            for filename in os.listdir(self.docker_volume_path):
                    
                # Check if the UUID for the file exists in the mapping
                file_uuid = self.file_uuid_mapping.get(filename)
                if file_uuid:
                    # If UUID exists, add it to the result
                    files_info.append({
                        "filename": filename,
                        "uuid": file_uuid
                    })
                else:
                    self.logger.warning(f"No UUID found for file: {filename}")

            self.logger.info("Successfully listed %d files.", len(files_info))
            return files_info
        except Exception as e:
            self.logger.exception("Error listing files: %s", e)
            return {"error": str(e)}
    
    async def extract_content_from_pdf(self, file: str) -> List[Document]:
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        chunks = splitter.split_documents(docs)  # Split documents into chunks
        self.logger.info("Successfully extracted and split PDF into %d chunks.", len(chunks))
        
        return chunks
    
    async def process_pdf(self, files: List[UploadFile]) -> List[Document]:
        """
        Process uploaded PDF files, save them in a Docker volume, and extract content chunks.
        If a file is already stored, skip the extraction.

        Args:
            files (List[UploadFile]): List of uploaded PDF files.
            retriever (Optional[str]): Selected retriever type. Defaults to None.

        Returns:
            List[Document]: A list of Document chunks extracted from the PDFs.
        """
        
        all_chunks = []
        file_uuid_mapping = {}
        self.logger.info("Starting to process PDF files.")

        for file in files:
            try:
                # Generate a unique ID for the PDF
                pdf_id = str(uuid.uuid4())
                file_uuid_mapping[file.filename] = pdf_id
                self.save_uuid_mapping()
                
                self.logger.info("Generated unique ID for file %s: %s", file.filename, pdf_id)

                # Path where the file will be saved inside the Docker container
                temp_file_path = os.path.join(self.docker_volume_path, file.filename)
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                # Check if the file already exists to avoid redundant processing
                if os.path.exists(temp_file_path):
                    self.logger.info(f"File {file.filename} already exists. Skipping processing.")
                    continue

                # Save the uploaded file to the Docker volume (inside the container)
                with open(temp_file_path, "wb") as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                self.logger.info("Saved file %s to %s", file.filename, temp_file_path)

                chunks = await self.extract_content_from_pdf(temp_file_path)
                
                # Assign metadata to chunks
                for chunk in chunks:
                    chunk.metadata["pdf_id"] = pdf_id
                    chunk.metadata["source_filename"] = file.filename

                all_chunks.extend(chunks)
                self.logger.info("File %s processed and indexed with %d chunks.", file.filename, len(chunks))
                
            except Exception as e:
                self.logger.exception("Error processing the PDF %s: %s", file.filename, e)
                raise RuntimeError(f"Failed to process PDF {file.filename}")

        self.logger.info("PDF processing completed successfully with a total of %d chunks extracted.", len(all_chunks))
        return all_chunks, file_uuid_mapping
    
    