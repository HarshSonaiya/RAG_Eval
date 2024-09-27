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
        """Sets up the user interface for the Streamlit app."""
        st.title("RAG Pipeline PDF Processing and Comparison")

        self.uploaded_pdfs = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        self.rag_model = st.selectbox("Choose the RAG model to process your query:", options=[
            "Hybrid Retriever", "HyDE Retriever", "Multiquery Retriever", "Dense Retriever", "All"
        ])
        self.user_query = st.text_area("Enter a query", height=100)

        if st.button("Submit"):
            self.process_request()

    def process_request(self):
        """Processes the request when the user submits the form."""
        if not self.uploaded_pdfs:
            st.warning("Please upload at least one PDF document.")
            return
        if not self.user_query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Processing your request..."):
            files = [('files', (uploaded_file.name, uploaded_file, 'application/pdf')) for uploaded_file in self.uploaded_pdfs]

            try:
                selected_rag_model = {
                    "Hybrid Retriever": "hybrid_rag",
                    "HyDE Retriever": "hyde_rag",
                    "Multiquery Retriever": "multiquery_rag",
                    "Dense Retriever": "dense_rag",
                    "All": "all"
                }.get(self.rag_model)
                response = requests.post(f"http://backend:9000/api/{selected_rag_model}", files=files, data={"query": self.user_query})

                if response.status_code == 200:
                    results = response.json()
                    self.display_results(results, selected_rag_model)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                self.logger.exception(f"An error occurred: {e}")
                
    def display_results(self, results, selected_rag_model):
        """Displays the results received from the backend."""

        if selected_rag_model == "all":
            models = ["hybrid", "hyde", "dense", "multiquery"]
            evaluations = {}

            # Extract LLM and Retriever evaluations for each model
            for model in models:
                result = results.get(model, {})
                llm_eval = result.get(f"{model}_rag_llm_eval", [])
                retriever_eval = result.get(f"{model}_rag_retriever_eval", [])
                evaluations[model] = {
                    "llm_eval": [metric.strip() for metric in llm_eval[0].split(",")] if llm_eval else [],
                    "retriever_eval": [metric.strip() for metric in retriever_eval[0].split(",")] if retriever_eval else [],
                    "response": result.get(f"{model}_rag_response", "No response available.")
                }

            # Display LLM evaluations in a tabular format
            st.subheader("LLM Evaluations (Side by Side)")
            llm_table = {
                "Model": [],
                "Helpfulness": [],
                "Correctness": [],
                "Coherence": [],
                "Complexity": [],
                "Verbosity": []
            }

            for model in models:
                llm_eval_data = evaluations[model]["llm_eval"]
                if llm_eval_data:
                    llm_table["Model"].append(model.capitalize())
                    for i, metric in enumerate(llm_eval_data):
                        category, value = metric.split(":")
                        llm_table[category.capitalize()].append(float(value))
                else:
                    # Fill with N/A for models without eval data
                    llm_table["Model"].append(model.capitalize())
                    llm_table["Helpfulness"].append("N/A")
                    llm_table["Correctness"].append("N/A")
                    llm_table["Coherence"].append("N/A")
                    llm_table["Complexity"].append("N/A")
                    llm_table["Verbosity"].append("N/A")

            st.table(llm_table)

            # Display Retriever evaluations in a tabular format
            st.subheader("Retriever Evaluations (Side by Side)")
            retriever_table = {
                "Model": [],
                "Helpfulness": [],
                "Correctness": [],
                "Coherence": [],
                "Complexity": [],
                "Verbosity": []
            }

            for model in models:
                retriever_eval_data = evaluations[model]["retriever_eval"]
                if retriever_eval_data:
                    retriever_table["Model"].append(model.capitalize())
                    for i, metric in enumerate(retriever_eval_data):
                        category, value = metric.split(":")
                        retriever_table[category.capitalize()].append(float(value))
                else:
                    # Fill with N/A for models without eval data
                    retriever_table["Model"].append(model.capitalize())
                    retriever_table["Helpfulness"].append("N/A")
                    retriever_table["Correctness"].append("N/A")
                    retriever_table["Coherence"].append("N/A")
                    retriever_table["Complexity"].append("N/A")
                    retriever_table["Verbosity"].append("N/A")

            st.table(retriever_table)

            # Display all RAG responses one after the other
            st.subheader("RAG Responses")
            for model in models:
                st.write(f"**Response from {model.capitalize()} RAG:**")
                st.write(evaluations[model]["response"])

        else:
            # Handle single model responses for other cases
            response = results.get(f"{selected_rag_model}_response", "No response available.")
            llm_eval = results.get(f"{selected_rag_model}_llm_eval", [])
            retriever_eval = results.get(f"{selected_rag_model}_retriever_eval", [])

            st.subheader(f"Response from {selected_rag_model.capitalize()} RAG")
            st.write(response)

            # Prepare evaluations
            llm_eval_data = [metric.strip() for metric in llm_eval[0].split(",")] if llm_eval else []
            retriever_eval_data = [metric.strip() for metric in retriever_eval[0].split(",")] if retriever_eval else []

            # Create two columns for displaying the evaluations side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("LLM Evaluation")
                if llm_eval_data:
                    st.table(llm_eval_data)
                else:
                    st.write("No LLM evaluation metrics available.")

            with col2:
                st.subheader("Retriever Evaluation")
                if retriever_eval_data:
                    st.table(retriever_eval_data)
                else:
                    st.write("No Retriever evaluation metrics available.")

# Initialize the RAGApp
rag_app = RAGApp()
