"""
================================================================================
   GIGASCALE RAG ENGINE PRO - ENTERPRISE EDITION v5.0
================================================================================
   Author: Achyut Anand Pandey (Automated Architecture)
   Copyright (c) 2026 Enterprise AI Solutions
   
   DESCRIPTION:
   A monolithic, high-availability Retrieval Augmented Generation (RAG) system 
   designed for multi-tenant, secure, and resilient document intelligence.
   
   ARCHITECTURE MODULES:
   1. Configuration & Secrets Management (Strict Validation)
   2. Telemetry & Observability (JSON Structured Logging)
   3. Document Ingestion Pipeline (Polyglot Loading, Semantic Splitting)
   4. Vector Persistence Layer (FAISS with Session Isolation)
   5. Cognitive Layer (GPT-4o Agent with Tool Usage)
   6. Session & Context Management (Conversation Windowing)
================================================================================
"""

import os
import shutil
import logging
import json
import time
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# --- THIRD PARTY DEPENDENCIES ---
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# LangChain Ecosystem
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableConfig

# ==============================================================================
# MODULE 1: CONFIGURATION MANAGEMENT
# ==============================================================================

load_dotenv()

class AppConfig(BaseModel):
    """Immutable Configuration Object for the RAG Engine."""
    
    # Azure Credentials
    AZURE_OPENAI_API_KEY: str = Field(..., description="The API Key for Azure OpenAI")
    AZURE_OPENAI_ENDPOINT: str = Field(..., description="The Base Endpoint URL")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-12-01-preview")
    
    # Model Deployments
    CHAT_MODEL_DEPLOYMENT: str = Field(default="gpt-4o")
    EMBEDDING_MODEL_DEPLOYMENT: str = Field(default="text-embedding-3-small")
    
    # System Constraints
    MAX_RECURSION_LIMIT: int = 5
    SESSION_STORAGE_PATH: str = "secure_enterprise_sessions"
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 250
    TEMPERATURE: float = 0.0

    class Config:
        frozen = True

    @classmethod
    def load(cls) -> 'AppConfig':
        """Factory method to load and validate env vars."""
        try:
            # specialized URL cleaner
            raw_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            clean_url = raw_url.strip().strip('"').strip("'").rstrip('/')
            if "/openai" in clean_url:
                clean_url = clean_url.split("/openai")[0]

            return cls(
                AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
                AZURE_OPENAI_ENDPOINT=clean_url,
                AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
            )
        except ValidationError as e:
            logging.critical(f"CONFIGURATION FATAL ERROR: {e}")
            raise RuntimeError("System configuration failed. Check .env file.")

# Global Config Instance
GLOBAL_CONFIG = AppConfig.load()

# ==============================================================================
# MODULE 2: TELEMETRY & OBSERVABILITY
# ==============================================================================

class LogLevel(Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Telemetry:
    """Structured JSON Logger for Observability Platforms."""
    
    @staticmethod
    def _log(level: LogLevel, message: str, context: Dict[str, Any] = None):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "message": message,
            "service": "RAG_ENGINE_CORE",
            "context": context or {}
        }
        # In a real system, this might push to Datadog/Splunk
        print(json.dumps(payload))

    @staticmethod
    def info(msg: str, **kwargs): Telemetry._log(LogLevel.INFO, msg, kwargs)
    @staticmethod
    def warn(msg: str, **kwargs): Telemetry._log(LogLevel.WARN, msg, kwargs)
    @staticmethod
    def error(msg: str, **kwargs): Telemetry._log(LogLevel.ERROR, msg, kwargs)
    @staticmethod
    def critical(msg: str, **kwargs): Telemetry._log(LogLevel.CRITICAL, msg, kwargs)

# ==============================================================================
# MODULE 3: DOCUMENT PROCESSING PIPELINE
# ==============================================================================

class DocumentProcessor:
    """Handles the extraction, cleaning, and chunking of raw files."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes artifacts, null bytes, and excessive whitespace."""
        text = text.replace('\x00', '') # Remove null bytes
        text = re.sub(r'\n\s*\n', '\n\n', text) # Merge multiple newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII trash
        return text.strip()

    @staticmethod
    def load_directory(directory_path: str) -> List[Any]:
        """Polyglot loader for PDF, DOCX, and TXT."""
        docs = []
        if not os.path.exists(directory_path):
            Telemetry.warn("Directory not found during load", path=directory_path)
            return []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            loader = None
            
            try:
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                
                if loader:
                    loaded_docs = loader.load()
                    # Post-process content
                    for d in loaded_docs:
                        d.page_content = DocumentProcessor.clean_text(d.page_content)
                        d.metadata["source_file"] = filename
                    docs.extend(loaded_docs)
                    Telemetry.info(f"Loaded file: {filename}", pages=len(loaded_docs))
                else:
                    Telemetry.warn(f"Skipping unsupported file: {filename}")
            
            except Exception as e:
                Telemetry.error(f"Failed to load file {filename}", error=str(e))
        
        return docs

    @staticmethod
    def chunk_documents(documents: List[Any]) -> List[Any]:
        """Splits documents using recursive semantic strategies."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=GLOBAL_CONFIG.CHUNK_SIZE,
            chunk_overlap=GLOBAL_CONFIG.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        splits = splitter.split_documents(documents)
        Telemetry.info(f"Chunking complete", original_pages=len(documents), chunks_created=len(splits))
        return splits

# ==============================================================================
# MODULE 4: VECTOR PERSISTENCE LAYER
# ==============================================================================

class VectorStoreManager:
    """Manages the lifecycle of FAISS indices per session."""

    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.base_path = GLOBAL_CONFIG.SESSION_STORAGE_PATH
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def get_session_db_path(self, session_id: str) -> str:
        return os.path.join(self.base_path, session_id)

    def create_index(self, session_id: str, splits: List[Any]):
        """Creates and persists a FAISS index for a session."""
        try:
            path = self.get_session_db_path(session_id)
            if os.path.exists(path):
                shutil.rmtree(path) # Enforce clean state
            
            vector_store = FAISS.from_documents(splits, self.embeddings)
            vector_store.save_local(path)
            Telemetry.info("Vector Index Created", session_id=session_id, path=path)
            return vector_store
        except Exception as e:
            Telemetry.critical("Vector Index Creation Failed", error=str(e))
            raise e

    def load_index(self, session_id: str):
        """Loads an existing FAISS index."""
        try:
            path = self.get_session_db_path(session_id)
            if not os.path.exists(path):
                Telemetry.warn("Index not found", session_id=session_id)
                return None
            
            vector_store = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            Telemetry.error("Failed to load index", session_id=session_id, error=str(e))
            return None

# ==============================================================================
# MODULE 5: COGNITIVE AGENT LAYER
# ==============================================================================

class AgentPersona(str, Enum):
    """Defines the behavioral mode of the agent."""
    DEFAULT = "default"
    INTERVIEWER = "interviewer"
    SUMMARIZER = "summarizer"
    CODING_EXPERT = "coding_expert"

class PromptFactory:
    """Generates rigorous System Prompts based on persona."""

    @staticmethod
    def get_system_prompt(persona: AgentPersona = AgentPersona.DEFAULT) -> str:
        base_rules = (
            "You are an elite RAG Agent. "
            "Your ONLY source of truth is the retrieved document context. "
            "Do NOT hallucinate. Do NOT use outside knowledge unless explicitly asked. "
            "If the answer is not in the context, state: 'I cannot find that info in the documents.'\n"
            "FORMATTING: Use plain text. No markdown. No #, **, etc."
        )

        if persona == AgentPersona.INTERVIEWER:
            return base_rules + "\nMODE: INTERVIEWER. Ask probing questions based on the resume. Be critical."
        elif persona == AgentPersona.SUMMARIZER:
            return base_rules + "\nMODE: SUMMARIZER. Be concise. Bullet points only."
        elif persona == AgentPersona.CODING_EXPERT:
            return base_rules + "\nMODE: CODING. Analyze code quality, security, and performance."
        else:
            return base_rules

class RAGBrain:
    """The central intelligence unit wrapping the LLM and Agent Executor."""

    def __init__(self):
        self._init_models()
        self.vector_manager = VectorStoreManager(self.embeddings)
        self.sessions: Dict[str, ConversationBufferMemory] = {}

    def _init_models(self):
        """Initializes Azure Clients with retry logic."""
        Telemetry.info("Initializing Cognitive Models...")
        self.llm = AzureChatOpenAI(
            azure_deployment=GLOBAL_CONFIG.CHAT_MODEL_DEPLOYMENT,
            api_version=GLOBAL_CONFIG.AZURE_OPENAI_API_VERSION,
            azure_endpoint=GLOBAL_CONFIG.AZURE_OPENAI_ENDPOINT,
            api_key=GLOBAL_CONFIG.AZURE_OPENAI_API_KEY,
            temperature=GLOBAL_CONFIG.TEMPERATURE,
            max_retries=3
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=GLOBAL_CONFIG.EMBEDDING_MODEL_DEPLOYMENT,
            api_version=GLOBAL_CONFIG.AZURE_OPENAI_API_VERSION,
            azure_endpoint=GLOBAL_CONFIG.AZURE_OPENAI_ENDPOINT,
            api_key=GLOBAL_CONFIG.AZURE_OPENAI_API_KEY
        )

    def _get_memory(self, session_id: str) -> ConversationBufferMemory:
        """Retrieves or creates conversation memory for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        return self.sessions[session_id]

    def ingest(self, session_id: str, upload_dir: str) -> Dict[str, Any]:
        """Public Ingestion Entrypoint."""
        start_time = time.time()
        
        # 1. Load
        raw_docs = DocumentProcessor.load_directory(upload_dir)
        if not raw_docs:
            return {"status": "failed", "reason": "No valid documents found"}
        
        # 2. Split
        chunks = DocumentProcessor.chunk_documents(raw_docs)
        
        # 3. Index
        self.vector_manager.create_index(session_id, chunks)
        
        # 4. Reset Memory (New Context)
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            
        duration = time.time() - start_time
        return {
            "status": "success", 
            "pages": len(raw_docs), 
            "chunks": len(chunks),
            "duration_seconds": round(duration, 2)
        }

    def query(self, session_id: str, user_input: str, persona: str = "default") -> Dict[str, str]:
        """Public Query Entrypoint with Dynamic Tooling."""
        
        # 1. Validate Session
        vector_store = self.vector_manager.load_index(session_id)
        if not vector_store:
            return {"answer": "Session expired or empty. Please upload documents."}

        # 2. Build Tool
        @tool
        def dynamic_retriever(query: str):
            """Searches the session's specific document database."""
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)
                if not docs: return "No results."
                return "\n\n".join([f"[Page {d.metadata.get('page','?')}] {d.page_content}" for d in docs])
            except Exception as e:
                return f"Retrieval Error: {e}"

        tools = [dynamic_retriever]

        # 3. Build Prompt
        try:
            target_persona = AgentPersona(persona)
        except ValueError:
            target_persona = AgentPersona.DEFAULT
            
        system_msg = PromptFactory.get_system_prompt(target_persona)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 4. Execute Agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self._get_memory(session_id),
            verbose=True,
            max_iterations=GLOBAL_CONFIG.MAX_RECURSION_LIMIT,
            handle_parsing_errors=True
        )

        try:
            Telemetry.info("Executing Agent", session_id=session_id, input=user_input)
            response = executor.invoke({"input": user_input})
            
            # 5. Output Sanitization
            raw_text = response.get("output", "Error: No output")
            clean_text = TextFormatter.clean(raw_text)
            
            return {"answer": clean_text}
            
        except Exception as e:
            Telemetry.error("Agent Execution Failed", error=str(e))
            return {"answer": "I encountered a critical error processing your request."}

class TextFormatter:
    """Utilities for cleaning LLM output for frontend display."""
    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r'#+\s*', '', text) # Headers
        text = re.sub(r'\*\*', '', text)  # Bold
        return text.strip()

# ==============================================================================
# MODULE 6: GLOBAL INSTANTIATION & EXPORTS
# ==============================================================================

# Singleton Pattern to prevent multiple model initializations
try:
    _ENGINE_INSTANCE = RAGBrain()
except Exception as e:
    Telemetry.critical("FAILED TO START RAG ENGINE", error=str(e))
    _ENGINE_INSTANCE = None

def build_database(upload_dir: str, session_id: str = "default") -> Dict[str, Any]:
    """Wrapper for external calls."""
    if not _ENGINE_INSTANCE: return {"status": "error", "message": "System Offline"}
    return _ENGINE_INSTANCE.ingest(session_id, upload_dir)

def get_answer(query: str, session_id: str = "default") -> Dict[str, str]:
    """Wrapper for external calls."""
    if not _ENGINE_INSTANCE: return {"answer": "System Offline"}
    return _ENGINE_INSTANCE.query(session_id, query)