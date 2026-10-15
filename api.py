import time
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag import create_qa_chain   # ✅ NON CAMBIA MAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ⚠️ CAMBIA PER OGNI PROGETTO: titolo dell'API
app = FastAPI(title="AutoVeloce RAG API", version="1.0.0")

# Crea la chain una sola volta all'avvio
qa_chain = create_qa_chain()   # ✅ NON CAMBIA MAI

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Method: {request.method} | Path: {request.url.path} | Time: {process_time:.4f}s | Status: {response.status_code}")
    return response

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def root():
    # ⚠️ CAMBIA PER OGNI PROGETTO: messaggio di benvenuto
    return {"status": "online", "message": "AutoVeloce RAG API is running!"}

@app.post("/ask", response_model=Answer)
def ask_question(q: Question):
    logger.info(f"Received question: {q.question}")
    start = time.time()
    result = qa_chain.invoke({"query": q.question})
    elapsed = time.time() - start
    sources = [doc.metadata.get('source', 'unknown') for doc in result['source_documents']]
    logger.info(f"Answered in {elapsed:.2f}s | Sources: {sources}")
    return Answer(answer=result['result'], sources=sources)