"""
ENTERPRISE API GATEWAY
----------------------
Serves the RAG Engine via FastAPI with strict validation and error handling.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import logging
import time
from typing import List, Optional

# Import the Engine
from rag_engine import build_database, get_answer

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_GATEWAY")

# --- App Definition ---
app = FastAPI(
    title="Agentic RAG Enterprise API",
    description="Secure, Session-Based Document Intelligence API",
    version="2.1.0"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Performance monitoring middleware."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Path: {request.url.path} | Time: {process_time:.4f}s")
    return response

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default-session"

class APIResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[dict] = None

# --- Routes ---

@app.get("/", response_model=APIResponse)
def health_check():
    """System Health Check Endpoint."""
    return {
        "status": "online",
        "message": "Enterprise RAG System is Operational",
        "data": {"version": "2.1.0"}
    }

@app.post("/upload")
async def upload_docs(
    files: list[UploadFile] = File(...), 
    session_id: str = Form("default-mobile-session")
):
    """
    Secure Document Upload Handler.
    Creates a unique temporary directory for the session and triggers indexing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # 1. Secure Directory Creation
        upload_dir = f"temp_data/{session_id}"
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir) # Clean slate
        os.makedirs(upload_dir, exist_ok=True)

        # 2. File Save Loop
        saved_files = []
        for file in files:
            # Basic validation
            if not file.filename.endswith(('.pdf', '.docx', '.txt')):
                continue 
                
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

        if not saved_files:
             raise HTTPException(status_code=400, detail="No valid file types uploaded (PDF/DOCX only).")

        # 3. Trigger Engine Indexing
        logger.info(f"Triggering ingestion for Session {session_id} with {len(saved_files)} files.")
        status_message = build_database(upload_dir, session_id=session_id)
        
        return {
            "message": f"Successfully processed {len(saved_files)} files.", 
            "rag_status": status_message
        }

    except Exception as e:
        logger.critical(f"Upload Critical Fail: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Chat Endpoint.
    Routes the user query to the specific agent instance for their session.
    """
    try:
        if not request.query.strip():
             raise HTTPException(status_code=400, detail="Query cannot be empty.")

        response = get_answer(request.query, session_id=request.session_id)
        return response
        
    except Exception as e:
        logger.error(f"Chat Execution Fail: {e}")
        raise HTTPException(status_code=500, detail="Agent Execution Failed.")