import os
import shutil
import tempfile
from typing import List
import logging
from fastapi import UploadFile
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.globals import set_debug

set_debug(True)

logger = logging.getLogger(__name__)

class PdfService:
    def __init__(self, logger):
        pass

    async def extract_content_from_pdf(self, file: UploadFile) -> List[Document]:
        """
        Extract and split content from a PDF file into chunks.

        Args:
            file (str): Path to the PDF file.

        Returns:
            List[Document]: A list of Documents containing various attributes
            like page_content, metadata, etc. extracted from the PDF.
        """
        try:
            logger.info("Extracting content from PDF file: %s", file.filename)

            # Save the uploaded file temporarily in memory
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name  # Get the temporary file path

            # Load and process the PDF using the PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            total_word_count = sum(len(doc.page_content.split()) for doc in docs)

            # Determine adaptive chunk size based on word count
            base_chunk_size = 900
            density_threshold = 1.5
            chunk_size = (
                base_chunk_size // 2
                if total_word_count / base_chunk_size > density_threshold
                else base_chunk_size
            )

            # Set adaptive overlap proportional to chunk size
            min_overlap = 50  # Minimum overlap
            max_overlap = 200  # Maximum overlap
            overlap_ratio = 0.2  # Adjust overlap as 20% of chunk size
            overlap_size = max(
                min_overlap, min(int(chunk_size * overlap_ratio), max_overlap)
            )

            logger.info(
                "Adaptive chunk size: %d, Adaptive overlap: %d (total word count: %d)",
                chunk_size,
                overlap_size,
                total_word_count,
            )

            # Initialize the text splitter with specified chunk size and overlap
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=150
            )
            chunks = splitter.split_documents(docs)

            logger.info(
                "Successfully extracted and split PDF '%s' into %d chunks.",
                file.filename,
                len(chunks),
            )

            return chunks
        except Exception as e:
            logger.exception(
                "Error extracting content from PDF file '%s': %s", file.filename, e
            )
            raise e
        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
