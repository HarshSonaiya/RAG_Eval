import json
import logging

import requests
import streamlit as st

logger = logging.getLogger("streamlit")


class RAGApp:
    """
    A class to manage the RAG Pipeline PDF Processing and Comparison application.
    """

    def __init__(self):
        """
        Initializes the RAGApp, sets up the logger, and configures the Streamlit app.
        """
        self.file_list = []
        self.brain_id = None
        self.setup_ui()

    def setup_ui(self):
        """Sets up the user interface for the Streamlit app."""

        st.title("RAG Pipeline PDF Processing and Comparison")

        # Section to create a new brain
        st.subheader("Create a New Brain")
        brain_name = st.text_input("Enter a name for the new brain:")

        if st.button("Create Brain"):
            if brain_name.strip():
                self.create_new_brain(brain_name.strip())

        # After creating a new brain, show the updated list of brains
        st.header("Select a Brain")
        brains = self.fetch_brain_list()

        if isinstance(brains, list) and brains:
            # Display the list of brains
            brain_names = [
                brain["brain_name"] for brain in brains if "brain_name" in brain
            ]
            selected_brain = st.selectbox(
                "Select a brain to work with:", options=set(brain_names)
            )

            # Find the corresponding brain ID for the selected brain
            self.brain_id = next(
                (
                    brain["brain_id"]
                    for brain in brains
                    if brain["brain_name"] == selected_brain
                ),
                None,
            )

            if self.brain_id:
                self.handle_pdf_upload_and_query()
            else:
                st.warning("Please select a valid brain to proceed.")
        else:
            st.warning("No brains available. Please create a brain first.")

        # New Section: Evaluation
        st.header("Evaluate Responses")
        evaluation_file = st.file_uploader(
            "Upload Test Dataset (XLSX format)", type=["xlsx"]
        )

        if st.button("Run Evaluation"):
            if evaluation_file:
                self.run_evaluation(evaluation_file)
            else:
                st.warning("Please upload a test dataset.")

    def run_evaluation(self, evaluation_file):
        """Runs the evaluation using the uploaded CSV file."""
        with st.spinner("Running evaluation..."):
            try:
                files = {
                    "test_csv": (evaluation_file.name, evaluation_file, "text/csv")
                }
                response = requests.post(
                    "http://backend:9000/api/evaluate", files=files
                )
                response = response.json()
                if response["success"]:
                    self.display_evaluation_results(response["data"])
                else:
                    st.error(response['message'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception(f"Evaluation error: {e}")

    def create_new_brain(self, brain_name):
        """Creates a new brain by sending a request to the backend."""
        with st.spinner(f"Creating brain: {brain_name}..."):
            try:
                response = requests.post(
                    "http://backend:9000/api/create-brain", data={"brain_name": brain_name}
                )
                response = response.json()
                if response["success"]:
                    st.success(response["message"])
                else:
                    st.warning(response["message"])
            except Exception as e:
                st.error(f"Could not connect to the backend: {str(e)}")

    def fetch_brain_list(self):
        """Fetches the list of available brains."""
        try:
            response = requests.get("http://backend:9000/api/list-brains")
            response = response.json()

            if response["success"]:
                return response["data"]
            else:
                st.error(response['message'])
        except Exception as e:
            st.error(f"Could not connect to the backend: {str(e)}")
            return []

    def handle_pdf_upload_and_query(self):
        """Handles PDF upload and query input when a brain is selected."""

        st.header("Upload PDF Documents")
        uploaded_pdfs = st.file_uploader(
            "Upload PDF documents", type="pdf", accept_multiple_files=True
        )

        if st.button("Process Uploaded PDFs"):
            if uploaded_pdfs:
                self.process_uploaded_pdfs(uploaded_pdfs)
                self.file_list = self.fetch_file_list()
            else:
                st.info("No PDFs uploaded. Moving on to the query step.")

        st.header("Select PDFs for Query")
        self.file_list = (
            self.fetch_file_list() if not self.file_list else self.file_list
        )

        if self.file_list:
            filenames = [file_info["file_name"] for file_info in self.file_list]
            filenames.insert(0, "All PDFs")

            selected_pdfs = st.multiselect(
                "Select a PDF:",
                options=filenames,
            )

            if "All PDFs" in selected_pdfs:
                selected_pdf_data = [
                    {
                        "file_name": file_info["file_name"],
                        "file_id": file_info["file_id"],
                    }
                    for file_info in self.file_list
                ]
                st.info("Query will be searched across all PDFs.")
            else:
                selected_pdf_data = [
                    {
                        "file_name": file_info["file_name"],
                        "file_id": file_info["file_id"],
                    }
                    for file_info in self.file_list
                    if file_info["file_name"] in selected_pdfs
                ]
        else:
            selected_pdfs = []
            st.warning("No PDFs available. Please upload some first.")

        st.header("Enter Your Query")
        user_query = st.text_area("Enter your question here", height=100)
        rag_model = st.selectbox(
            "Choose the RAG model to process your query:",
            options=[
                "Hybrid Retriever",
                "HyDE Pipeline with Dense Retriver",
                "Sparse Retriever",
                "Dense Retriever",
                "All",
            ],
        )

        if st.button("Submit Query"):
            if selected_pdf_data and user_query.strip():
                self.process_request(selected_pdf_data, user_query, rag_model)
            elif not selected_pdf_data:
                st.warning("Please select at least one PDF.")
            else:
                st.warning("Please enter a query.")

    def process_uploaded_pdfs(self, uploaded_files):
        """Processes uploaded PDFs: assigns unique IDs, uploads to the backend, and updates status."""
        with st.spinner("Uploading and processing PDFs..."):
            files = [
                ("files", (uploaded_file.name, uploaded_file, "application/pdf"))
                for uploaded_file in uploaded_files
            ]

            try:
                response = requests.post(
                    f"http://backend:9000/api/{self.brain_id}/upload", files=files
                )
                response = response.json()
                if response["success"]:
                    st.success(response["message"])
                else:
                    st.error(response["message"])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception(f"An error occurred: {e}")

    def fetch_file_list(self):
        """Fetches the list of uploaded PDFs from the backend."""
        with st.spinner("Fetching the list of available PDFs..."):
            try:
                response = requests.get(
                    f"http://backend:9000/api/{self.brain_id}/list-files"
                )
                response = response.json()
                if response["success"]:
                    return response["data"]
                else:
                    st.error(response['message'])
                    return []
            except Exception as e:
                st.error(f"Could not connect to the backend: {str(e)}")
                return []

    def process_request(self, selected_pdfs, user_query, rag_model):
        """Processes the request when the user submits the form."""

        with st.spinner("Processing your request..."):
            try:
                selected_rag_model = {
                    "Hybrid Retriever": "hybrid",
                    "HyDE Pipeline with Dense Retriver": "hyde",
                    "Dense Retriever": "dense",
                    "Sparse Retriever": "sparse",
                    "All": "all",
                }.get(rag_model)

                payload = {
                    "query": user_query,
                    "selected_pdfs": selected_pdfs,
                }

                response = requests.post(
                    f"http://backend:9000/api/{self.brain_id}/{selected_rag_model}",
                    json=payload,
                )
                response = response.json()

                if response["success"]:
                    self.display_results(response["data"], selected_rag_model)
                else:
                    st.error(response['message'])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception(f"An error occurred: {e}")

    def display_results(self, results, selected_rag_model):
        """Displays the results received from the backend."""

        st.header("Query Results")
        if selected_rag_model == "all":
            st.subheader("Results from All RAG Models")
            models = ["hybrid", "hyde", "dense"]
            evaluations = {}

            # Extract LLM and Retriever evaluations for each model
            for model in models:
                result = results.get(model, {})
                llm_eval = result.get(f"{model}_rag_llm_eval", [])
                retriever_eval = result.get(f"{model}_rag_retriever_eval", [])
                evaluations[model] = {
                    "llm_eval": (
                        [metric.strip() for metric in llm_eval[0].split(",")]
                        if llm_eval
                        else []
                    ),
                    "retriever_eval": (
                        [metric.strip() for metric in retriever_eval[0].split(",")]
                        if retriever_eval
                        else []
                    ),
                    "response": result.get(
                        f"{model}_rag_response", "No response available."
                    ),
                    "retriever_response": result.get(
                        f"{model}_retriever_response", "No response available."
                    ),
                }

            # Display LLM evaluations in a tabular format
            st.subheader("LLM Evaluations (Side by Side)")
            llm_table = {
                "Model": [],
                "Helpfulness": [],
                "Correctness": [],
                "Coherence": [],
                "Complexity": [],
                "Verbosity": [],
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
                "Verbosity": [],
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
                st.markdown(
                    f"""
                            <span style="font-size: 20px;">
                            **Response from {model.capitalize()} RAG: **
                            </span>
                        """,
                    unsafe_allow_html=True,
                )
                st.write(evaluations[model]["response"])
                st.markdown(
                    f"""
                            <span style="font-size: 20px;">
                            **Response from {model.capitalize()} RAG Retriever:**
                            </span>
                        """,
                    unsafe_allow_html=True,
                )
                st.write(evaluations[model]["retriever_response"])

        else:
            # Handle single model responses for other cases
            response = results.get(
                f"{selected_rag_model}_rag_response", "No response available."
            )
            retrieved_context = results.get(
                f"{selected_rag_model}_retriever_response", "No response available."
            )
            llm_eval = results.get(f"{selected_rag_model}_llm_eval", [])
            retriever_eval = results.get(f"{selected_rag_model}_retriever_eval", [])

            st.subheader(f"Response from {selected_rag_model.capitalize()} RAG: ")
            st.write(response)

            st.subheader(
                f"Response from {selected_rag_model.capitalize()} RAG Retriever: "
            )
            st.write(retrieved_context)

            # Prepare evaluations
            llm_eval_data = (
                [metric.strip() for metric in llm_eval[0].split(",")]
                if llm_eval
                else []
            )
            retriever_eval_data = (
                [metric.strip() for metric in retriever_eval[0].split(",")]
                if retriever_eval
                else []
            )

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
