# RAG_Eval: Evaluation of Retrievers in RAG Pipelines with Free Alternatives

***RAG_Eval*** is a project designed to test various retrievers that support response generation in Retrieval-Augmented Generation (RAG) pipelines. 
It leverages the NeMoTron family of models for cost-free response evaluation, providing an alternative to paid tools like RAGAs (which require OpenAI's paid API).

---

## 📋 OVERVIEW

The project aims to compare the performance of different retrievers in RAG pipelines using evaluation metrics such as **Helpfulness, Correctness, Coherence, Complexity, and Verbosity.** 
Instead of relying on paid APIs like **OpenAI's** for evaluation using **RAGAs**, RAG_Eval integrates the NeMoTron models for evaluation, offering a free and open-source alternative 
for performance analysis.

---

## ✨ Features

**1. Workspace Management:**
  - Create and manage workspaces, where each workspace contains a specific group of user uploaded PDFs for streamlined querying.
    
**2. Workspace and PDF Selection:**
  - Select a specific workspace for querying and narrowing down the scope by choosing single, multiple, or all PDFs within that workspace.
    
**3. Retriever Testing:**
  - Compare and analyze performance across Dense, Hybrid, Multi-query, and other retrievers.
    
**4. Free Evaluation Models:**
  - Uses NVIDIA's NeMoTron APIs, eliminating dependence on paid services.
    
**5. Comprehensive Metrics:**
  - Evaluate responses using key metrics:
    - **Helpfulness**
    - **Correctness**
    - **Coherence**
    - **Complexity**
    - **Verbosity.**
      
**6. Customizable Pipelines:** 
  - Easily extend or modify pipelines to test additional retrievers or evaluation metrics.

---

## 🚀 Installation

### Step1: Clone the Repository
```bash 
git clone https://github.com/HarshSonaiya/RAG_Eval.git
cd RAG_Eval
```

---

### Step2: Install Docker and Docker Compose and verify installations
```bash
docker --version
docker-compose --version
```

---
    
### Step3: Create and Configure the .env File

   a. Copy the provided `.env.example` file to create a new `.env` file:
       ```bash
       cp .env.example .env
       ```
    
   b. Open the .env file in a text editor of your choice
   c. Update the placeholder values with your configuration details.

---

### Step4: Build and Start the Services
```bash
docker-compose up --build
```

---

### Step5: Verfiy Setup

  a. Check Running Containers
      ```
      docker ps
      ```
  b. Access the application via the provided URL (e.g., `http://localhost:8501` for Streamlit).

---

## 📚 API Documentation

1. `/create-brain:` Create Workspace for uploading PDFs.
2. `/list-brains:` List down all the created workspaces.
3. `/upload:` Upload PDF Files for indexing.
4. `/list-files:` List down all the uploaded PDF Files in the selected workspace.
5. `/hybrid_rag:` Process query with hybrid retriever in traditional RAG pipeline.
6. `/hybde_rag:` Process query with dense retriever in HyDE RAG pipeline.
7. `/dense_rag:` Process query with dense retriever in traditional rag pipeline.
8. `/all:` Process all the retrievers.

---

## 🏆 Acknowledgements

This project was made possible through the invaluable guidance and mentorship of:

- **Rahul Parmar** (https://github.com/therahulparmar)
- **Keval Dekivadiya** (https://github.com/kevaldekivadiya2415)

---
