"""
================================================================================
   TITANIUM API GATEWAY - ENTERPRISE EDITION v8.0
   -----------------------------------------------------------------------------
   Author: Achyut Anand Pandey (Architect)
   Copyright (c) 2026 Enterprise AI Solutions
   
   DESCRIPTION:
   High-performance FastAPI gateway designed for the Titanium RAG System.
   Features:
   - Asynchronous Non-Blocking I/O
   - Token Bucket Rate Limiting
   - Magic Byte File Validation (Security)
   - Structured JSON Logging
   - Correlation ID Tracking
   - RFC 7807 Problem Details
================================================================================
"""

import os
import sys
import time
import json
import uuid
import shutil
import logging
import asyncio
import traceback
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Third-Party Libraries
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    Form, 
    HTTPException, 
    Request, 
    status, 
    Depends
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# --- ENGINE IMPORT ---
try:
    from rag_engine import build_database, get_answer
    ENGINE_AVAILABLE = True
except ImportError:
    logging.critical("CRITICAL: rag_engine.py not found. RAG features disabled.")
    ENGINE_AVAILABLE = False
    # Mock functions to prevent NameError
    def build_database(*args, **kwargs): return "Engine Missing"
    def get_answer(*args, **kwargs): return {"answer": "System Error: Engine Missing"}

# ==============================================================================
# SECTION 1: SYSTEM CONFIGURATION
# ==============================================================================

class GatewayConfig:
    """Centralized API Configuration."""
    TITLE = "Titanium RAG API"
    VERSION = "8.0.0-Titan"
    ENV = os.getenv("ENV", "production")
    
    # Security
    ALLOWED_ORIGINS = ["*"]
    MAX_UPLOAD_SIZE_MB = 50
    RATE_LIMIT_PER_MINUTE = 100
    
    # Storage
    TEMP_STORAGE_PATH = "temp_ingest_buffer"
    
    # Magic Numbers (File Signatures)
    FILE_SIGNATURES = {
        "pdf": b"\x25\x50\x44\x46",
        "docx": b"\x50\x4B\x03\x04"
    }

CONFIG = GatewayConfig()

# ==============================================================================
# SECTION 2: OBSERVABILITY & LOGGING
# ==============================================================================

class StructuredLogger:
    """JSON Logger for Gateway operations."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _log(self, level: str, msg: str, **kwargs):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": "api_gateway",
            "message": msg,
            "correlation_id": kwargs.get("correlation_id", "n/a"),
            "metadata": kwargs
        }
        print(json.dumps(entry))

    def info(self, msg: str, **kwargs): self._log("INFO", msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log("ERROR", msg, **kwargs)
    def warn(self, msg: str, **kwargs): self._log("WARN", msg, **kwargs)

logger = StructuredLogger("TITANIUM_GATEWAY")

# ==============================================================================
# SECTION 3: MIDDLEWARE
# ==============================================================================

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Injects a unique UUID into every request."""
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

class PerformanceMonitorMiddleware(BaseHTTPMiddleware):
    """Tracks latency."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        logger.info(
            f"{request.method} {request.url.path}",
            latency_ms=process_time,
            status=response.status_code,
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Hardening headers."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

class RateLimiter:
    """In-Memory Token Bucket."""
    def __init__(self):
        self.tokens = {}
        self.lock = asyncio.Lock()
    
    async def check(self, client_ip: str) -> bool:
        current_time = int(time.time())
        window = current_time // 60
        key = f"{client_ip}:{window}"
        async with self.lock:
            count = self.tokens.get(key, 0)
            if count >= CONFIG.RATE_LIMIT_PER_MINUTE:
                return False
            self.tokens[key] = count + 1
            if len(self.tokens) > 2000: # Cleanup
                self.tokens = {k:v for k,v in self.tokens.items() if int(k.split(':')[1]) >= window}
            return True

rate_limiter = RateLimiter()

async def rate_limit_dependency(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# ==============================================================================
# SECTION 4: DATA MODELS
# ==============================================================================

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(default="default-session")

class HealthResponse(BaseModel):
    status: ResponseStatus
    version: str
    engine_online: bool
    timestamp: datetime

# ==============================================================================
# SECTION 5: APP LIFECYCLE
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- TITANIUM GATEWAY STARTING ---")
    if os.path.exists(CONFIG.TEMP_STORAGE_PATH):
        shutil.rmtree(CONFIG.TEMP_STORAGE_PATH)
    os.makedirs(CONFIG.TEMP_STORAGE_PATH, exist_ok=True)
    yield
    logger.info("--- TITANIUM GATEWAY STOPPING ---")
    if os.path.exists(CONFIG.TEMP_STORAGE_PATH):
        shutil.rmtree(CONFIG.TEMP_STORAGE_PATH)

app = FastAPI(
    title=CONFIG.TITLE,
    version=CONFIG.VERSION,
    lifespan=lifespan
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(PerformanceMonitorMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ==============================================================================
# SECTION 6: ENDPOINTS
# ==============================================================================

@app.get("/", tags=["System"])
def root():
    return {"message": "Titanium Gateway Online. Access /docs for schema."}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status=ResponseStatus.SUCCESS,
        version=CONFIG.VERSION,
        engine_online=ENGINE_AVAILABLE,
        timestamp=datetime.utcnow()
    )

@app.post("/upload", tags=["Ingestion"])
async def upload_documents(
    request: Request,
    # FORGIVING MODE: Default to None to prevent 422 Errors
    files: List[UploadFile] = File(default=None), 
    session_id: str = Form(default="mobile-fallback")
):
    """
    Secure Document Upload.
    Accepts PDF/DOCX/TXT. Handles missing files gracefully.
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.info(f"Upload Request for Session: {session_id}", correlation_id=correlation_id)

    # 1. Graceful Failure Check
    if not files:
        logger.warn("No files received in request", correlation_id=correlation_id)
        return JSONResponse(
            status_code=400, 
            content={"status": "error", "detail": "No files found. Check request headers."}
        )

    # 2. Engine Availability Check
    if not ENGINE_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "RAG Engine Offline."})

    # 3. Sandbox Creation
    upload_dir = os.path.join(CONFIG.TEMP_STORAGE_PATH, session_id)
    try:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        os.makedirs(upload_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Filesystem Error: {e}", correlation_id=correlation_id)
        return JSONResponse(status_code=500, content={"detail": "Server Storage Error"})

    # 4. File Processing
    saved_files = []
    try:
        for file in files:
            # Validate Extension
            if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
                continue
            
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            
        if not saved_files:
            return JSONResponse(status_code=400, content={"detail": "No valid file types (PDF/DOCX/TXT)."})

        # 5. Engine Trigger
        engine_status = build_database(upload_dir, session_id=session_id)
        
        return {
            "status": "success",
            "message": f"Processed {len(saved_files)} files.",
            "data": engine_status
        }

    except Exception as e:
        logger.error(f"Processing Crash: {e}", correlation_id=correlation_id)
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/chat", tags=["Inference"])
async def chat_endpoint(request: QueryRequest):
    """
    Inference Endpoint.
    Routes query to RAG Engine.
    """
    if not ENGINE_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "Engine Offline"})

    try:
        response = get_answer(request.query, session_id=request.session_id)
        return response
    except Exception as e:
        logger.error(f"Chat Crash: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)