import streamlit as st
import requests
from config.logging_config import LoggerFactory

# Initialize logger
logger = LoggerFactory()
logger = logger.get_logger("streamlit")

# Title of the app
st.title("RAG Pipeline PDF Processing and Comparison")

# Check if the PDFs were already uploaded in the session
if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = None
    st.session_state.uploaded_pdf_names = set()

# File uploader for PDFs
uploaded_pdfs = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

rag_model = st.selectbox(
    "Choose the RAG model to process your query:",
    options=["Hybrid Retriever", "HyDE Retriever", "Multiquery Retriever"]
)

rag_model_mapping = {
    "Hybrid Retriever": "hybrid_rag",
    "HyDE Retriever": "hyde_rag",
    "Multiquery Retriever": "multiquery_rag"
}

# Get the corresponding code name for the selected model
selected_rag_model = rag_model_mapping.get(rag_model)

# Store uploaded PDFs in session state to avoid resubmitting
if uploaded_pdfs:
    new_pdfs = [
        pdf for pdf in uploaded_pdfs
        if pdf.name not in st.session_state.uploaded_pdf_names
    ]
    if new_pdfs:
        st.session_state.uploaded_pdfs = new_pdfs
        st.session_state.uploaded_pdf_names.update([pdf.name for pdf in new_pdfs])
        logger.info(f"New PDFs uploaded: {[pdf.name for pdf in new_pdfs]}")
    else:
        logger.info("No new PDFs uploaded. Using previously uploaded PDFs.")

# Optional: User can also input a specific query to perform
user_query = st.text_area("Enter a query", height=100)

# Submit button
if st.button("Submit"):
    # Check if PDFs and query are available
    if not st.session_state.uploaded_pdfs:
        st.warning("Please upload at least one PDF document.")
        logger.warning("No PDF was uploaded by the user.")
    elif user_query.strip() == "":
        st.warning("Please enter a query.")
        logger.warning("No query was entered by the user.")
    else:
        # Code to send the request to the backend
        with st.spinner("Processing your request..."):
            # Prepare the files for the request
            files = [
                ('files', (uploaded_file.name, uploaded_file, 'application/pdf'))
                for uploaded_file in st.session_state.uploaded_pdfs
            ]

            try:
                # Send request to backend
                response = requests.post(
                    f"http://localhost:9000/api/{selected_rag_model}",
                    files=files,
                    data={"query": user_query}
                )
                logger.info("Request sent to FastAPI backend.")

                # Process the response
                if response.status_code == 200:
                    results = response.json()
                    logger.info("Successful response received from FastAPI backend.")

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

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    logger.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception(f"An error occurred: {e}")
