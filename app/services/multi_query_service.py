from config.settings import settings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from config.logging_config import LoggerFactory  
from utils.llm_manager import LLMManager
from RAG_Eval.app.utils.collection import Collection

# Initialize logger using LoggerFactory
logger_factory = LoggerFactory()
logger = logger_factory.get_logger("pipeline")

class MultiQueryService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection = Collection(client)
        self.prompt_template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown: """
        # self.vector_store = QdrantVectorStore(
        #     client=self.client, 
        #     collection_name=settings.DENSE_COLLECTION, 
        #     embedding = settings.DENSE_EMBEDDING_MODEL,
        #     vector_name= "dense",
        # )        
        # self.retriever = self.vector_store.as_retriever(search_type="mmr")
        # Initialize ChatGroq as the LLM
        self.llm = LLMManager()
        
        # Create MultiQueryRetriever using the vector store and LLM

    def multi_query_search(self, query: str, limit=5):
        """
        Perform a multi-query search based on the provided queries using MultiQueryRetriever.
        """
        # Use the generate method to retrieve results
        results = self.multi_query_retriever.invoke(query)
        return results

    def generate_response(self, question: str, context: str):
        """
        Generate a response using the LLM based on the provided question and context.
        """
        formatted_prompt = self.prompt_template.format(question=question, context=context)

        response = self.llm.llm.invoke(formatted_prompt)
        return response
    

