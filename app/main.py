from fastapi import FastAPI
from controllers import pdf_controller

app = FastAPI()

# Include the PDF processing routes from the controller
app.include_router(pdf_controller.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Pipeline API"}
