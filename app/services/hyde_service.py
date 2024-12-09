import logging
from typing import List
from uuid import uuid4

from config.settings import settings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from utils.collection import Collection
from utils.llm_manager import LLMManager

# Initialize logger using LoggerFactory
logger = logging.getLogger("pipeline")


class HyDEService:

    def __init__(self, client: QdrantClient, llm_manager: LLMManager):
        self.client = client
        self.llm_manager = llm_manager
        self.prompt_template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question.If you are not provided with any extracted
        parts of the documments then try to generate an answer based on your knowledge and facts in your knowledge. Remember to provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown: """

    async def index_collection(self, chunks: List[Document]):
        """
        Index the given list of Document chunks into the Qdrant dense collection.
        """
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        self.vector_store.add_documents(documents=chunks, ids=uuids)

    def hyde_search(self, query: str):
        """
        Perform a HyDE search based on the provided query using QdrantVectorStore.
        """
        results = self.retriever.invoke(query)
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLMManager and prompt template.
        """
        formatted_prompt = self.prompt_template.format(
            question=question, context=context
        )

        response = self.llm_manager.llm.invoke(formatted_prompt)
        return response
