from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import logging
from rag_engine import build_database, get_answer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_SERVER")

app = FastAPI()

# Enable CORS so the Frontend can talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model for Chat (Includes Session ID)
class QueryRequest(BaseModel):
    query: str
    session_id: str 

@app.get("/")
def home():
    return {"status": "Secure Multi-User System Online"}

@app.post("/upload")
async def upload_docs(
    files: list[UploadFile] = File(...), 
    session_id: str = Form(...)  # <--- Catches the Badge from Frontend
):
    try:
        # Create a unique temporary folder for THIS user
        upload_dir = f"temp_data/{session_id}"
        
        # Clean up old files for this specific user if they exist
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        os.makedirs(upload_dir)

        saved_files = []
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

        # Tell the Brain to index these files into the User's private DB
        status_message = build_database(upload_dir, session_id=session_id)
        
        return {
            "message": f"Successfully processed {len(saved_files)} files.",
            "rag_status": status_message
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        # Pass the query AND the session_id to the engine
        response = get_answer(request.query, session_id=request.session_id)
        return response
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))