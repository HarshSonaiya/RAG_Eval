import os
from dotenv import load_dotenv


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

    def load_settings(self):
        """
        Loads the environment variables from the specified file.
        """
        load_dotenv(self.env_file)
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.GROQ_API_KEY =  os.getenv("GROQ_API_KEY")
        self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
        self.HYBRID_COLLECTION = os.getenv("HYBRID_COLLECTION_NAME")
        self.DENSE_COLLECTION = os.getenv("DENSE_COLLECTION_NAME")
        self.DENSE_EMBEDDING_MODEL = os.getenv("DENSE_MODEL")
        self.SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_MODEL")


settings = AppSettings()
