"""
ENTERPRISE RAG ENGINE v2.0
--------------------------
Author: Achyut Anand Pandey (Automated Refactor)
Date: January 2026
Description: 
    A high-availability, session-based Retrieval Augmented Generation (RAG) 
    orchestrator designed for robust document analysis.
    
    Features:
    - Multi-User Session Isolation
    - FAISS Vector Storage with Fallback Mechanisms
    - Recursive Text Splitting with Semantic Overlap
    - GPT-4o Tool-Calling Agent Architecture
    - Strict Output Sanitization
"""

import os
import shutil
import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# --- Third-Party Imports ---
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferMemory

# --- 1. SYSTEM CONFIGURATION ---
load_dotenv()

# Configure Enterprise Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [SESSION:%(threadName)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ENTERPRISE_RAG_CORE")

# Constants
BASE_STORAGE_PATH = "secure_user_sessions"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
MAX_HISTORY_BUFFER = 10
EMBEDDING_MODEL_DEPLOYMENT = "text-embedding-3-small"
CHAT_MODEL_DEPLOYMENT = "gpt-4o"

# --- 2. UTILITY CLASSES ---

class UrlSanitizer:
    """Helper class to ensure Azure endpoints are correctly formatted."""
    
    @staticmethod
    def clean(url: str) -> str:
        """
        Strips trailing slashes and removes internal paths from Azure Endpoint URLs.
        
        Args:
            url (str): Raw URL string from environment variables.
            
        Returns:
            str: Cleaned base URL.
        """
        if not url:
            return ""
        url = url.strip().strip('"').strip("'").rstrip('/')
        if "/openai" in url:
            url = url.split("/openai")[0]
        return url

class TextFormatter:
    """Handles output sanitization to strictly enforce plain text rules."""

    @staticmethod
    def clean_output(text: str) -> str:
        """
        Removes Markdown, bolding, and other rich text artifacts.
        
        Args:
            text (str): Raw LLM output.
            
        Returns:
            str: Plain text response.
        """
        # Remove Headers (###)
        text = re.sub(r'#+\s*', '', text)
        # Remove Bold/Italic (** or __ or *)
        text = re.sub(r'\*\*|__|\*', '', text)
        # Remove Link Brackets
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        return text.strip()

# --- 3. CORE SESSION MANAGER ---

class EnterpriseSessionManager:
    """
    Singleton Class managing the lifecycle of user sessions, 
    vector databases, and LLM connections.
    """
    
    def __init__(self):
        """Initializes the infrastructure and verifies cloud connectivity."""
        logger.info("Initializing Enterprise RAG System...")
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        try:
            self._load_credentials()
            self._init_azure_clients()
            self._ensure_storage_integrity()
            logger.info("System Initialization Complete. Ready to serve.")
        except Exception as e:
            logger.critical(f"FATAL SYSTEM ERROR during initialization: {e}")
            raise e

    def _load_credentials(self):
        """Loads and validates environment variables."""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.endpoint = UrlSanitizer.clean(os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Missing critical Azure Credentials in environment.")

    def _init_azure_clients(self):
        """Sets up the LLM and Embedding connections."""
        logger.info(f"Connecting to Azure OpenAI at {self.endpoint}...")
        
        self.llm = AzureChatOpenAI(
            azure_deployment=CHAT_MODEL_DEPLOYMENT,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            temperature=0.0, # Zero temperature for maximum factual consistency
            max_retries=3
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=EMBEDDING_MODEL_DEPLOYMENT,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            max_retries=3
        )

    def _ensure_storage_integrity(self):
        """Ensures the local file system is ready for secure storage."""
        if not os.path.exists(BASE_STORAGE_PATH):
            os.makedirs(BASE_STORAGE_PATH)
            logger.info(f"Created secure storage directory: {BASE_STORAGE_PATH}")

    def get_session_path(self, session_id: str) -> str:
        """Generates a secure path for a specific user's vector database."""
        return os.path.join(BASE_STORAGE_PATH, session_id)

    def ingest_documents(self, session_id: str, upload_dir: str) -> str:
        """
        Orchestrates the full document ingestion pipeline:
        Load -> Split -> Embed -> Index -> Save.
        """
        logger.info(f"Starting ingestion for Session ID: {session_id}")
        
        # 1. Validation
        if not os.path.exists(upload_dir):
            logger.warning("Upload directory does not exist.")
            return "Upload failed: Directory not found."

        # 2. Document Loading
        documents = []
        files_processed = 0
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            try:
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    files_processed += 1
                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                    files_processed += 1
                else:
                    logger.warning(f"Skipping unsupported file type: {filename}")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")

        if not documents:
            logger.warning("No valid text content extracted from uploaded files.")
            return "No readable documents found."

        # 3. Text Splitting (Recursive Semantic)
        logger.info(f"Splitting {len(documents)} pages of text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            keep_separator=True
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} vector chunks.")

        # 4. Vector Database Generation
        try:
            user_db_path = self.get_session_path(session_id)
            
            # Ensure path exists before write
            if not os.path.exists(user_db_path):
                os.makedirs(user_db_path)
            
            vector_store = FAISS.from_documents(splits, self.embeddings)
            vector_store.save_local(user_db_path)
            
            # 5. Memory Initialization
            self.sessions[session_id] = {
                "memory": ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True,
                    output_key="output"
                ),
                "last_active": datetime.now()
            }
            
            logger.info(f"Ingestion successful for Session {session_id}")
            return f"Successfully indexed {files_processed} documents ({len(documents)} pages)."
            
        except Exception as e:
            logger.critical(f"Vector Database Failure: {e}")
            raise RuntimeError(f"Critical Indexing Error: {e}")

    def create_retrieval_tool(self, session_id: str):
        """Dynamic Tool Factory: Creates a search tool bound to a specific user's DB."""
        
        user_db_path = self.get_session_path(session_id)

        @tool
        def document_retriever(query: str):
            """
            CRITICAL TOOL: Searches the user's uploaded documents for information.
            Input should be a specific search query related to the user's question.
            """
            if not os.path.exists(user_db_path):
                return "ERROR: No database found. Ask user to upload a file."
            
            try:
                # Load secure index
                vector_store = FAISS.load_local(
                    user_db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5} # Fetch top 5 relevant chunks
                )
                docs = retriever.invoke(query)
                
                if not docs:
                    return "No relevant information found in the documents."
                
                # Format context with page numbers
                context_str = "\n---\n".join(
                    [f"[Page {d.metadata.get('page', 'Unknown')}] {d.page_content}" for d in docs]
                )
                return context_str
                
            except Exception as e:
                logger.error(f"Retrieval Exception: {e}")
                return f"System Error during retrieval: {str(e)}"
        
        return document_retriever

    def execute_agent(self, session_id: str, user_query: str) -> Dict[str, Any]:
        """
        Runs the full Agentic Loop: 
        User Query -> Agent Thought -> Tool Call -> Observation -> Final Answer.
        """
        # 1. Security Check
        session_data = self.sessions.get(session_id)
        user_db_path = self.get_session_path(session_id)

        if not session_data or not os.path.exists(user_db_path):
            logger.warning(f"Unauthorized access attempt for Session {session_id}")
            return {"answer": "Session Expired. Please reload the page and upload your document again."}

        # 2. Setup Agent Tools
        retrieval_tool = self.create_retrieval_tool(session_id)
        tools = [retrieval_tool]

        # 3. DEFINE THE STRICT SYSTEM PROMPT (User Specified)
        system_prompt_content = (
            "You are an elite, enterprise-grade Retrieval-Augmented Generation (RAG) agent.\n\n"
            "Your sole responsibility is to answer user queries strictly by analyzing the retrieved documents.\n"
            "You must never rely on prior knowledge, assumptions, or external information.\n\n"
            "PRIMARY OBJECTIVE:\n"
            "- Provide accurate, factual, and document-grounded answers based only on retrieved content.\n"
            "- If the required information is not present in the documents, explicitly say so.\n\n"
            "STRICT OPERATION RULES:\n"
            "1. You must ALWAYS use the document_retriever tool before answering any user query.\n"
            "2. You are forbidden from answering without first consulting retrieved documents.\n"
            "3. You must not hallucinate, infer, assume, or fabricate information.\n"
            "4. You must not merge external knowledge with document content.\n"
            "5. If multiple documents conflict, report the conflict clearly instead of choosing one.\n"
            "6. If the answer is partially present, state what is found and what is missing.\n\n"
            "OUTPUT FORMAT RULES:\n"
            "- Use plain text only.\n"
            "- Do NOT use markdown.\n"
            "- Do NOT use symbols like *, **, #.\n"
            "- Use simple dashes (-) for lists only when necessary.\n"
            "- Be concise, professional, and direct.\n\n"
            "FAILURE MODE:\n"
            "If no relevant information is retrieved, respond exactly with:\n"
            "\"I cannot find this information in the provided documents.\"\n\n"
            "You are a factual analysis system, not a creative assistant.\n"
            "Accuracy is more important than completeness."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 4. Create Agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=session_data["memory"],
            verbose=True, # Logs thoughts to console
            max_iterations=5, # Prevents infinite loops
            handle_parsing_errors=True
        )

        # 5. Execute
        try:
            logger.info(f"Agent executing query for Session {session_id}")
            result = executor.invoke({"input": user_query})
            
            # 6. Post-Process Output
            raw_answer = result.get("output", "No response generated.")
            clean_answer = TextFormatter.clean_output(raw_answer)
            
            return {"answer": clean_answer}
            
        except Exception as e:
            logger.error(f"Agent Crash: {e}")
            return {"answer": "I encountered an internal processing error. Please check the system logs."}

# --- 4. GLOBAL SINGLETON INSTANCE ---
# This ensures we don't re-initialize models on every API call
try:
    rag_engine = EnterpriseSessionManager()
except Exception as e:
    logger.critical("Failed to instantiate RAG Engine.")
    rag_engine = None

# --- 5. PUBLIC API HANDLERS ---
# These are the functions imported by main.py

def build_database(upload_dir: str, session_id: str = "default"):
    """Public wrapper for document ingestion."""
    if not rag_engine:
        return "System Offline: Engine failed to initialize."
    return rag_engine.ingest_documents(session_id, upload_dir)

def get_answer(query: str, session_id: str = "default"):
    """Public wrapper for query execution."""
    if not rag_engine:
        return {"answer": "System Offline: Engine failed to initialize."}
    return rag_engine.execute_agent(session_id, query)