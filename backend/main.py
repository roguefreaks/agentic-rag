from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import time
import fitz  # <--- This is PyMuPDF
from rag_engine import get_answer, build_database

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
# Store usage: { "127.0.0.1": { "files": 2, "pages": 10, "reset_time": 123456789 } }
user_limits = {}

LIMIT_FILES = 4
LIMIT_PAGES = 15
RESET_TIME = 7200  # 2 Hours

class QueryRequest(BaseModel):
    query: str

def check_rate_limit(ip: str):
    current_time = time.time()
    
    # Initialize or Reset
    if ip not in user_limits or current_time > user_limits[ip]["reset_time"]:
        user_limits[ip] = {
            "files": 0,
            "pages": 0,
            "reset_time": current_time + RESET_TIME
        }
    
    usage = user_limits[ip]
    
    if usage["files"] >= LIMIT_FILES:
        return True, "FILE_LIMIT_REACHED"
    if usage["pages"] >= LIMIT_PAGES:
        return True, "PAGE_LIMIT_REACHED"
        
    return False, None

@app.get("/")
def read_root():
    return {"status": "System Online", "model": "o3-mini", "library": "PyMuPDF"}

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host
    
    # 1. Check Rate Limit
    is_limited, reason = check_rate_limit(client_ip)
    if is_limited:
        raise HTTPException(status_code=429, detail=reason)

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Count Pages using PyMuPDF (Fast)
    try:
        doc = fitz.open(file_path)
        num_pages = doc.page_count
        doc.close()
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Invalid PDF or Corrupt File")

    # 3. Update Usage
    user_limits[client_ip]["files"] += 1
    user_limits[client_ip]["pages"] += num_pages
    
    # 4. Re-check limits
    if user_limits[client_ip]["pages"] > LIMIT_PAGES:
        os.remove(file_path) # Reject file
        user_limits[client_ip]["files"] -= 1 # Revert count
        raise HTTPException(status_code=429, detail="PAGE_LIMIT_EXCEEDED")

    # Indexing
    status = build_database(UPLOAD_DIR)
    
    return {
        "filename": file.filename, 
        "pages": num_pages,
        "status": "Indexed", 
        "usage": user_limits[client_ip]
    }

@app.post("/chat")
def chat(request: QueryRequest):
    response = get_answer(request.query)
    return response

@app.get("/files")
def list_files():
    if not os.path.exists(UPLOAD_DIR):
        return []
    files = os.listdir(UPLOAD_DIR)
    return [{"name": f, "size": f"{os.path.getsize(os.path.join(UPLOAD_DIR, f))/1024:.2f} KB"} for f in files]

@app.delete("/files/{filename}")
def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        build_database(UPLOAD_DIR)
        return {"status": "Deleted"}
    raise HTTPException(status_code=404, detail="File not found")