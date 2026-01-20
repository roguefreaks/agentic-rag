import os
import shutil
import time
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Core Engine
# We use a try-except block so the server starts even if the engine has a hiccup
try:
    from rag_engine import build_database, get_answer
    ENGINE_READY = True
except ImportError:
    ENGINE_READY = False

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_GATEWAY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default-session"

@app.get("/")
def health_check():
    return {"status": "ONLINE", "mode": "FORGIVING"}

@app.post("/upload")
async def upload_documents(
    # FIX: Default to None so the server doesn't throw 422 if files are missing
    files: List[UploadFile] = File(default=None), 
    session_id: str = Form(default="mobile-fallback") 
):
    try:
        # 1. Debugging Check
        if not files:
            logger.warning(f"Request received for Session {session_id} but NO FILES found.")
            return JSONResponse(
                status_code=400, 
                content={"detail": "No files received. Check 'Content-Type' header in frontend."}
            )

        # 2. Create Sandbox
        upload_dir = f"temp_data/{session_id}"
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        os.makedirs(upload_dir, exist_ok=True)

        # 3. Save Files
        saved_files = []
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

        # 4. Index
        if ENGINE_READY:
            status = build_database(upload_dir, session_id=session_id)
        else:
            status = "Engine Offline - Files Saved Only"
        
        return {
            "status": "success",
            "message": f"Processed {len(saved_files)} files.",
            "rag_status": status
        }

    except Exception as e:
        logger.error(f"Upload Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        if not ENGINE_READY:
             return JSONResponse(status_code=503, content={"detail": "Engine not ready."})
        
        response = get_answer(request.query, session_id=request.session_id)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})