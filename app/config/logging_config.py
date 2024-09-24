import logging
import os


class LoggerFactory:

    def __init__(self):
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)

        # Configure logging (done once in the constructor)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Handlers for each module's log file
        self.streamlit_handler = logging.FileHandler("logs/streamlit.log")
        self.streamlit_handler.setLevel(logging.DEBUG)

        self.server_handler = logging.FileHandler("logs/server.log")
        self.server_handler.setLevel(logging.DEBUG)

        self.pipeline_handler = logging.FileHandler("logs/pipeline.log")
        self.pipeline_handler.setLevel(logging.DEBUG)

    def get_logger(self, name):
        # Define logger for each part of the application
        logger = logging.getLogger(name)

        # Add appropriate handler based on the name
        if "streamlit" in name:
            if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == self.streamlit_handler.baseFilename for handler in logger.handlers):
                logger.addHandler(self.streamlit_handler)
        elif "server" in name:
            if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == self.server_handler.baseFilename for handler in logger.handlers):
                logger.addHandler(self.server_handler)
        elif "pipeline" in name:
            if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == self.pipeline_handler.baseFilename for handler in logger.handlers):
                logger.addHandler(self.pipeline_handler)

        return logger
