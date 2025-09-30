import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.model_loader import load_dotenv
from config.config_loader import load_config

class ModelLoader:
    """
    A utility class to load embedding models and LLM models.
    """

    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config=load_config()

    def _validate_env(self):
        """
        validate necessary environment variables.
        """
        required_vars = ["GOOGLE_API_KEY"]
        self.GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        

    def load_embeddings(self):
        """
        Load and return embedding model.
        """
        print("Loading Embedding model")
        model_name = self.config["embedding_model"]["model_name"]
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=self.GEMINI_API_KEY)

        

    def load_llm(self):
        """
        Load and return LLM model
        """
        print("Loading LLM model")
        model_name = self.config["llm"]["model_name"]
        return ChatGoogleGenerativeAI(model=model_name, google_api_key = self.GEMINI_API_KEY)