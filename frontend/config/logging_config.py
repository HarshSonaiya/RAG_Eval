import logging
import os


class LoggerFactory:
    """
    A factory class for creating and configuring loggers for different modules in the application.
    """

    def __init__(self):
        """
        Initializes the LoggerFactory and sets up log directories and handlers.
        """
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)

        # Configure logging (done once in the constructor)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Handlers for each module's log file
        self.handlers = {
            "streamlit": self.create_file_handler("logs/streamlit.log"),
            "server": self.create_file_handler("logs/server.log"),
            "pipeline": self.create_file_handler("logs/pipeline.log"),
            "pdf_processing": self.create_file_handler("logs/pdf_processing.log"),
            "Qdrant": self.create_file_handler("logs/Qdrant.log")
        }

    def create_file_handler(self, file_path):
        """
        Creates a file handler for logging.

        Args:
            file_path (str): The path to the log file.

        Returns:
            logging.FileHandler: The file handler for the specified log file.
        """
        handler = logging.FileHandler(file_path)
        handler.setLevel(logging.DEBUG)
        return handler

    def get_logger(self, name):
        """
        Retrieves a logger for the specified module name.

        Args:
            name (str): The name of the module for which to get a logger.

        Returns:
            logging.Logger: A logger instance configured for the specified module.
        """
        logger = logging.getLogger(name)

        # Add the appropriate handler based on the module name
        for key, handler in self.handlers.items():
            if key in name and not any(isinstance(h, logging.FileHandler) and h.baseFilename == handler.baseFilename for h in logger.handlers):
                logger.addHandler(handler)

        return logger
