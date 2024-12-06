from langchain_groq import ChatGroq
from config.settings import settings
import logging
# Initialize logger using LoggerFactory
logger = logging.getLogger("pipeline")

class LLMManager:
    def __init__(self):
        self.model_name = settings.LLM_NAME
        self.api_key = settings.GROQ_API_KEY 
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        logger.info("LLM Initialized Successfully")
        return ChatGroq(
            temperature=0.5, 
            model_name=self.model_name,
            max_tokens=3500
        )
    
    def rerank_docs(self, documents, query):
        # Extract content from retrieved documents
        document_texts = [doc.payload['content'] for doc in documents if doc] 

        # Create pairs of query and document
        pairs = [(query, doc_text) for doc_text in document_texts]
    
        # Use the Cross-Encoder to predict relevance scores for the pairs
        scores = settings.CROSS_ENCODER_MODEL.predict(pairs)

        logger.info(f"Retrieved documents and their scores:{len(pairs)}- {scores} - {type(scores)}")
        
        # Rank the documents based on the cross-encoder scores
        ranked_documents = [doc for _, doc in sorted(zip(scores, documents), reverse=True, key=lambda x: x[0])]

        # Select top 4 documents based on the reranked scores
        top_4_documents = ranked_documents[:4]
        logger.info("Document Reranked Successfully")

        return top_4_documents

