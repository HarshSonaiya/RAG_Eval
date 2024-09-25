from langchain_groq import ChatGroq
from app.config.settings import settings


class LLMManager:
    def __init__(self, model_name="llama3-8b-8192", api_key=None):
        self.model_name = model_name
        self.api_key = settings.GROQ_API_KEY 
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        return ChatGroq(temperature=0.3, model_name=self.model_name, api_key=self.api_key)
