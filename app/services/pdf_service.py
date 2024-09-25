import os
import shutil
from typing import List
from langchain.docstore.document import Document
from fastapi import UploadFile

class PdfService:
    def __init__(self, processor, logger):
        self.processor = processor
        self.logger = logger
        # Directory path inside the Docker volume (mapped to the host)
        self.docker_volume_path = "/data/raw"  # This is the path inside the Docker container

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

        for file in files:
            try:
                # Path where the file will be saved inside the Docker container
                temp_file_path = os.path.join(self.docker_volume_path, file.filename)
                self.logger.info(f"Received file: {file.filename}")

                # Ensure the directory exists inside the Docker volume
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                # Check if the file already exists to avoid redundant processing
                if not os.path.exists(temp_file_path):
                    # Save the uploaded file to the Docker volume (inside the container)
                    with open(temp_file_path, "wb") as temp_file:
                        shutil.copyfileobj(file.file, temp_file)

                    # Extract chunks from the saved PDF file
                    chunks = self.processor.extract_content_from_pdf(temp_file_path)
                    all_chunks.extend(chunks)
                    self.logger.info(f"File {file.filename} processed and indexed.")
                else:
                    self.logger.info(f"File {file.filename} already exists in volume, skipping processing.")

            except Exception as e:
                self.logger.exception(f"Error processing the PDF {file.filename}: {e}")
                raise RuntimeError(f"Failed to process PDF {file.filename}")

        self.logger.info("PDF processing completed successfully.")
        return all_chunks
