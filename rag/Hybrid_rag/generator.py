import os
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

class LLMManager:
    def __init__(self, model_name="llama3-8b-8192", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        return ChatGroq(temperature=0.3, model_name=self.model_name, api_key=self.api_key)

class LLMChainManager:
    def __init__(self, llm_manager):
        self.llm = llm_manager.llm
        self.prompt_template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown: """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["question", "context"])
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def answer_query(self, query: str, combined_context: str):

        response = self.llm_chain.invoke({"question": query, "context": combined_context})['text']
        return response, combined_context

