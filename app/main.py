from fastapi import FastAPI
from routes.routes import router

app = FastAPI()

# Include the PDF processing routes from the controller
app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Pipeline API"}
