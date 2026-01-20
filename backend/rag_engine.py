"""
================================================================================
   TITAN COMPANION RAG ENGINE v5.0
================================================================================
   Author: Achyut Anand Pandey
   Description:
   The computational core for the Titan API Gateway.
   This module handles:
   - Polyglot Document Loading (PDF/DOCX/TXT)
   - Recursive Semantic Splitting
   - FAISS Vector Persistence (Session Isolated)
   - GPT-4o Agentic Reasoning
================================================================================
"""

import os
import shutil
import logging
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- THIRD PARTY DEPENDENCIES ---
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# --- CONFIGURATION ---
load_dotenv()

# Setup Logger to match Titan Main
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TITAN_ENGINE")

# Constants
BASE_STORAGE_PATH = "secure_titan_sessions"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

class TitanEngine:
    """Singleton Core for RAG Operations."""
    
    def __init__(self):
        self._init_environment()
        self.sessions = {}
        logger.info("Titan Engine Core Initialized.")

    def _init_environment(self):
        """Load Azure Credentials Securely."""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().strip('"').rstrip('/')
        if "/openai" in endpoint:
            endpoint = endpoint.split("/openai")[0]
        self.endpoint = endpoint
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not self.api_key or not self.endpoint:
            logger.critical("CRITICAL: Azure Credentials Missing in .env")
            raise ValueError("Azure API Key and Endpoint are required.")

        # Initialize Models
        try:
            self.llm = AzureChatOpenAI(
                azure_deployment=CHAT_MODEL,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                temperature=0.0,
                max_retries=3
            )
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=EMBEDDING_MODEL,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key
            )
        except Exception as e:
            logger.critical(f"Model Connect Failed: {e}")
            raise e

    def get_session_path(self, session_id: str) -> str:
        return os.path.join(BASE_STORAGE_PATH, session_id)

    def ingest(self, session_id: str, upload_dir: str) -> Dict[str, Any]:
        """Ingest documents into a session-isolated Vector DB."""
        if not os.path.exists(upload_dir):
            return {"status": "failed", "detail": "Upload directory missing"}

        # 1. Load
        documents = []
        for filename in os.listdir(upload_dir):
            path = os.path.join(upload_dir, filename)
            try:
                if filename.lower().endswith(".pdf"):
                    documents.extend(PyPDFLoader(path).load())
                elif filename.lower().endswith(".docx"):
                    documents.extend(Docx2txtLoader(path).load())
                elif filename.lower().endswith(".txt"):
                    documents.extend(TextLoader(path).load())
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

        if not documents:
            return {"status": "failed", "detail": "No valid text extracted"}

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = splitter.split_documents(documents)

        # 3. Index & Save
        try:
            db_path = self.get_session_path(session_id)
            if os.path.exists(db_path):
                shutil.rmtree(db_path) # Clean slate for new uploads
                
            vector_store = FAISS.from_documents(splits, self.embeddings)
            vector_store.save_local(db_path)
            
            # Reset Memory
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
            
            logger.info(f"Session {session_id}: Indexed {len(documents)} pages into {len(splits)} chunks.")
            return {
                "status": "success",
                "files_processed": len(os.listdir(upload_dir)),
                "chunks_created": len(splits)
            }
        except Exception as e:
            logger.error(f"Indexing Failed: {e}")
            raise RuntimeError(f"Vector DB Creation Failed: {e}")

    def query(self, session_id: str, user_query: str) -> Dict[str, Any]:
        """Execute Agentic RAG Query."""
        db_path = self.get_session_path(session_id)
        
        if not os.path.exists(db_path):
            return {"answer": "I cannot access your documents. Please upload them again (Session Expired)."}

        # 1. Tool Definition
        @tool
        def document_search(q: str):
            """Search the uploaded documents for specific information."""
            try:
                vector_store = FAISS.load_local(
                    db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(q)
                if not docs: return "No relevant text found."
                return "\n\n".join([f"[Page {d.metadata.get('page','?')}] {d.page_content}" for d in docs])
            except Exception as e:
                return f"Search Error: {e}"

        tools = [document_search]

        # 2. Strict System Prompt
        system_prompt = (
            "You are a strict, factual RAG Assistant.\n"
            "1. ALWAYS use the document_search tool to find answers.\n"
            "2. If the info is not in the documents, say 'I cannot find this information'.\n"
            "3. Do not invent facts.\n"
            "4. Format output as plain text (no markdown symbols like ## or **)."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 3. Execution
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.sessions[session_id],
            verbose=True,
            max_iterations=4,
            handle_parsing_errors=True
        )

        try:
            res = executor.invoke({"input": user_query})
            
            # Clean Output
            clean_text = re.sub(r'\*\*|__', '', res["output"]).strip()
            return {"answer": clean_text}
            
        except Exception as e:
            logger.error(f"Agent Failure: {e}")
            return {"answer": "I encountered an error while thinking. Please try again."}

# --- SINGLETON EXPORT ---
try:
    _ENGINE = TitanEngine()
except Exception:
    _ENGINE = None

def build_database(upload_dir, session_id="default"):
    if not _ENGINE: return "Engine Offline"
    return _ENGINE.ingest(session_id, upload_dir)

def get_answer(query, session_id="default"):
    if not _ENGINE: return {"answer": "Engine Offline"}
    return _ENGINE.query(session_id, query)