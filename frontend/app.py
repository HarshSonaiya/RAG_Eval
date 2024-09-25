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
            options=["Hybrid Retriever", "HyDE Retriever", "Multiquery Retriever"]
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
                    "Multiquery Retriever": "multiquery_rag"
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
        result_hybrid = results.get("hybrid_response", "No hybrid retriever response")
        hybrid_llm_eval = results.get("hybrid_llm_eval", "No Response")
        retriever_llm_eval = results.get("hybrid_retriever_eval", "No Response")

        result_hyde = results.get("hyde_response", "No HyDE retriever response")
        hyde_llm_eval = results.get("hyde_llm_eval", "No Response")
        hyde_retriever_llm_eval = results.get("hyde_retriever_eval", "No Response")

        # Display results
        st.subheader("Response from Hybrid Retriever")
        st.write(result_hybrid)
        st.subheader("Evaluation of Hybrid Retriever Response")
        st.subheader("LLM Evaluation")
        st.write(hybrid_llm_eval[0]["content"])
        st.subheader("Retriever Evaluation")
        st.write(retriever_llm_eval[0]["content"])

        st.subheader("Response from HyDE Retriever")
        st.write(result_hyde)
        st.subheader("Evaluation of HyDE Retriever Response")
        st.subheader("LLM Evaluation")
        st.write(hyde_llm_eval[0]["content"])
        st.subheader("Retriever Evaluation")
        st.write(hyde_retriever_llm_eval[0]["content"])