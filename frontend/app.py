import streamlit as st
import requests
from config.logging_config import LoggerFactory


class RAGApp:
    """
    A class to manage the RAG Pipeline PDF Processing and Comparison application.
    """

    def __init__(self):
        """
        Initializes the RAGApp, sets up the logger, and configures the Streamlit app.
        """
        self.logger = LoggerFactory().get_logger("streamlit")
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface for the Streamlit app.
        """
        st.title("RAG Pipeline PDF Processing and Comparison")

        # Check if the PDFs were already uploaded in the session
        if 'uploaded_pdfs' not in st.session_state:
            st.session_state.uploaded_pdfs = None
            st.session_state.uploaded_pdf_names = set()

        self.uploaded_pdfs = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True
        )

        self.rag_model = st.selectbox(
            "Choose the RAG model to process your query:",
            options=["Hybrid Retriever", "HyDE Retriever", "Multiquery Retriever", "Dense Retriever", "All"]
        )

        # Store uploaded PDFs in session state to avoid resubmitting
        if self.uploaded_pdfs:
            self.handle_uploaded_files()

        # Optional: User can also input a specific query to perform
        self.user_query = st.text_area("Enter a query", height=100)

        # Submit button
        if st.button("Submit"):
            self.process_request()

    def handle_uploaded_files(self):
        """
        Handles the logic for storing uploaded PDF files in session state.
        """
        new_pdfs = [
            pdf for pdf in self.uploaded_pdfs
            if pdf.name not in st.session_state.uploaded_pdf_names
        ]
        if new_pdfs:
            st.session_state.uploaded_pdfs = new_pdfs
            st.session_state.uploaded_pdf_names.update([pdf.name for pdf in new_pdfs])
            self.logger.info(f"New PDFs uploaded: {[pdf.name for pdf in new_pdfs]}")
        else:
            self.logger.info("No new PDFs uploaded. Using previously uploaded PDFs.")

    def process_request(self):
        """
        Processes the request when the user submits the form.
        """
        # Check if PDFs and query are available
        if not st.session_state.uploaded_pdfs:
            st.warning("Please upload at least one PDF document.")
            self.logger.warning("No PDF was uploaded by the user.")
            return
        
        if self.user_query.strip() == "":
            st.warning("Please enter a query.")
            self.logger.warning("No query was entered by the user.")
            return

        # Code to send the request to the backend
        with st.spinner("Processing your request..."):
            # Prepare the files for the request
            files = [
                ('files', (uploaded_file.name, uploaded_file, 'application/pdf'))
                for uploaded_file in st.session_state.uploaded_pdfs
            ]

            try:
                selected_rag_model = {
                    "Hybrid Retriever": "hybrid_rag",
                    "HyDE Retriever": "hyde_rag",
                    "Multiquery Retriever": "multiquery_rag",
                    "Dense Retriever": "dense_rag",
                    "All": "all"
                }.get(self.rag_model)

                # Send request to backend
                response = requests.post(
                    f"http://localhost:9000/api/{selected_rag_model}",
                    files=files,
                    data={"query": self.user_query}
                )
                self.logger.info("Request sent to FastAPI backend.")

                # Process the response
                self.handle_response(response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                self.logger.exception(f"An error occurred: {e}")

    def handle_response(self, response):
        """
        Handles the response from the backend.

        Args:
            response (requests.Response): The response object from the backend request.
        """
        if response.status_code == 200:
            results = response.json()
            self.logger.info("Successful response received from FastAPI backend.")

            # Display results based on the selected RAG model
            self.display_results(results)

        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            self.logger.error(f"Error: {response.status_code} - {response.text}")

    def display_results(self, results):
        """
        Displays the results received from the backend.

        Args:
            results (dict): The results returned from the backend.
        """
        if self.rag_model == "All":
            # Display results for all models
            self.display_model_results(results, "Hybrid Retriever", "hybrid")
            self.display_model_results(results, "HyDE Retriever", "hyde")
            self.display_model_results(results, "Multiquery Retriever", "multiquery")
            self.display_model_results(results, "Dense Retriever", "dense")
        else:
            # Display results for the selected model only
            model_key = self.rag_model.lower().replace(" ", "_")
            self.display_model_results(results, self.rag_model, model_key)

    def display_model_results(self, results, model_name, model_key):
        """
        Displays results for a specific RAG model.

        Args:
            results (dict): The results returned from the backend.
            model_name (str): The name of the model.
            model_key (str): The key used to fetch results for the model.
        """
        response_key = f"{model_key}_response"
        llm_eval_key = f"{model_key}_llm_eval"
        retriever_eval_key = f"{model_key}_retriever_eval"

        result = results.get(response_key, f"No {model_name} response")
        llm_eval = results.get(llm_eval_key, "No Response")
        retriever_eval = results.get(retriever_eval_key, "No Response")

        # Display results
        st.subheader(f"Response from {model_name}")
        st.write(result)

        if isinstance(llm_eval, list) and llm_eval:
            st.subheader(f"Evaluation of {model_name} Response")
            st.subheader("LLM Evaluation")
            st.write(llm_eval[0]["content"])
        
        if isinstance(retriever_eval, list) and retriever_eval:
            st.subheader("Retriever Evaluation")
            st.write(retriever_eval[0]["content"])

# Initialize the RAGApp
rag_app = RAGApp()
