import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder


class AppSettings:
    """
    A class to manage application settings loaded from environment variables.
    """

    def __init__(self, env_file=".env"):
        """
        Initializes the AppSettings by loading environment variables from the specified file.

        Args:
            env_file (str): The path to the environment file (default is ".env").
        """
        self.env_file = env_file
        self.load_settings()
        self.initialize_models()

    def load_settings(self):
        """
        Loads the environment variables from the specified file.
        """
        load_dotenv(self.env_file)
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.LLM_NAME = os.getenv("GROQ_LLM_NAME")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
        self.QDRANT_RECORD_STORE = os.getenv("QDRANT_RECORDS_STORE")
        self.CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME")
        self.DENSE_EMBEDDING_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")
        self.SPARSE_EMBEDDING_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME")

    def initialize_models(self):

        self.DENSE_EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=self.DENSE_EMBEDDING_MODEL_NAME
        )
        self.SPARSE_EMBEDDING_MODEL = SparseTextEmbedding(
            model_name=self.SPARSE_EMBEDDING_MODEL_NAME
        )
        self.CROSS_ENCODER_MODEL = CrossEncoder(
            model_name=self.CROSS_ENCODER_MODEL_NAME
        )


settings = AppSettings()
