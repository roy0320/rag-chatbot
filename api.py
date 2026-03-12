from fastapi import FastAPI

app = FastAPI(title="RAG Chatbot API")

@app.get("/")
def root():
    return {"message": "RAG Chatbot API 運行中"}