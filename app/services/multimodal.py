import io

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from app import logger
from app.apis.v1.parser.schema import (Chunk, Coordinates, CoordinateSystem,
                                       EmbeddingInfo, EmbeddingModelParams,
                                       KnowledgeParams)
from app.constants.constants import (COMBINE_TEXT_UNDER_N_CHARS,
                                     MAX_CHARACTERS, NEW_AFTER_N_CHARS,
                                     OVERLAP)
from app.core.helper.s3.download_doc import DownloadDocument


class PDFParser:
    def __init__(self) -> None:
        self.download_document = DownloadDocument()

    def _extract_documents_from_pdf(self, in_memory_file: io.BytesIO) -> list:
        """Extracts documents from PDF file.

        Args:
            in_memory_file (io.BytesIO): In-memory file-like object containing the PDF file.

        Returns:
            list: List of extracted document chunks.
        """
        try:
            raw_pdf_elements = partition_pdf(file=in_memory_file)
            return chunk_by_title(
                raw_pdf_elements,
                max_characters=MAX_CHARACTERS,
                overlap=OVERLAP,
                multipage_sections=True,
                new_after_n_chars=NEW_AFTER_N_CHARS,
                combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS,
            )
        except Exception as exc:
            logger.exception(
                f"Error occurred while extracting documents from PDF: {exc}"
            )
            raise

    def process_document(
        self,
        enterprise_uuid: str,
        brain_uuid: str,
        knowledge_uuid: str,
        knowledge_params: KnowledgeParams,
        embedding_params: EmbeddingModelParams
    ) -> list:
        """Processes the PDF document."""
        try:
            in_memory_file, status = self.download_document.download_document(
                knowledge_path=knowledge_params.knowledge_path,
            )
            logger.debug(
                f"Extracting chunks from PDF started -> {knowledge_params.knowledge_path}"
            )
            if not status:
                raise

            chunks = self._extract_documents_from_pdf(in_memory_file=in_memory_file)
            logger.debug(
                f"Extracting chunks from PDF Done -> {knowledge_params.knowledge_path}"
            )
            parsed_formatted_documents = []

            # Iterate through documents and set is_last_document flag
            for doc_idx, doc in enumerate(chunks, start=1):
                is_last_document = doc_idx == len(chunks)

                coordinates = []
                for elem in doc.metadata.orig_elements:
                    coordinate = Coordinates(
                        page_number=elem.metadata.page_number,
                        coordinate_points=list(elem.metadata.coordinates.points),
                        coordinate_system=CoordinateSystem(
                            width=elem.metadata.coordinates.system.width,
                            height=elem.metadata.coordinates.system.height,
                        ),
                    )
                    coordinates.append(coordinate)

                chunk = Chunk(
                    page_content=doc.text,
                    is_last_document=is_last_document,
                    brain_uuid=brain_uuid,
                    enterprise_uuid=enterprise_uuid,
                    knowledge_uuid=knowledge_uuid,
                    coordinates=coordinates,
                    knowledge=KnowledgeParams(
                        knowledge_name=knowledge_params.knowledge_name,
                        knowledge_type=knowledge_params.knowledge_type,
                        knowledge_path=knowledge_params.knowledge_path,
                    ),
                    embedding_info=EmbeddingInfo(name=embedding_params.name, provider=embedding_params.provider)
                )
                parsed_formatted_documents.append(chunk.model_dump())

            return parsed_formatted_documents

        except Exception as exc:
            logger.exception(f"Error occurred while parsing PDF documents: {exc}")
            raise