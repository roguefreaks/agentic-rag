import os
import logging
import re
import shutil
import uuid
from typing import List, Dict, Any

# Third-party imports
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# --- 1. CONFIGURATION & LOGGING SETUP ---
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger("RAG_ENGINE")

# Configuration Constants
BASE_DB_PATH = "user_sessions" # Changed to folder for multiple users
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_HISTORY = 10

# --- 2. ROBUST UTILITIES ---
def clean_azure_url(url: str) -> str:
    """Sanitizes Azure Endpoint URLs to prevent 404/400 errors."""
    if not url:
        return ""
    url = url.strip().strip('"').strip("'").rstrip('/')
    if "/openai" in url:
        url = url.split("/openai")[0]
    return url

def format_clean_output(text: str) -> str:
    """Removes Markdown symbols for a cleaner chat experience."""
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*|__', '', text)
    return text.strip()

# --- 3. THE SESSION MANAGER (Multi-User Architecture) ---
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Any] = {} # Store memory/DB for each user
        self._init_environment()
        self._init_models()
        logger.info("SessionManager System Initialized Successfully.")

    def _init_environment(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.endpoint = clean_azure_url(os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
        
        if not self.api_key or not self.endpoint:
            logger.critical("Missing Azure Credentials! System will fail.")
            raise ValueError("Azure API Key and Endpoint are required.")

    def _init_models(self):
        try:
            # Shared Models (Efficient)
            self.llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                temperature=0.0,
                max_retries=3
            )
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-small",
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                max_retries=3
            )
        except Exception as e:
            logger.critical(f"Model Initialization Failed: {e}")
            raise e

    def get_session_path(self, session_id: str):
        """Creates a unique folder path for this specific user."""
        return os.path.join(BASE_DB_PATH, session_id)

    def ingest_documents(self, session_id: str, upload_dir: str) -> str:
        """Processes files specifically for ONE session ID."""
        documents = []
        if not os.path.exists(upload_dir):
            return "No files found."

        # 1. Load Files
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            try:
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

        if not documents:
            return "No valid documents found to index."

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # 3. Vector Database (Unique per User)
        try:
            user_db_path = self.get_session_path(session_id)
            vector_store = FAISS.from_documents(splits, self.embeddings)
            vector_store.save_local(user_db_path)
            
            # Create fresh memory for this user
            self.sessions[session_id] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                "db_path": user_db_path
            }
            logger.info(f"Indexed {len(documents)} pages for Session {session_id}.")
            return f"Indexed {len(documents)} pages securely."
        except Exception as e:
            logger.error(f"Vector Store Failure: {e}")
            raise RuntimeError(f"Indexing Failed: {e}")

    def ask(self, session_id: str, query: str) -> Dict[str, Any]:
        """Runs the Agent specifically for the requested Session ID."""
        
        # 1. Retrieve User Context
        session_data = self.sessions.get(session_id)
        user_db_path = self.get_session_path(session_id)

        # Check if user has data (Security Check)
        if not session_data or not os.path.exists(user_db_path):
            return {"answer": "Please upload a document first to start a new session."}

        # 2. Define User-Specific Tool (The "Search This User's File" Tool)
        @tool
        def document_retriever(q: str):
            """
            Useful for finding specific information in the uploaded documents.
            Always use this tool first when asked about the candidate or file content.
            """
            try:
                # Load THIS user's specific DB
                vector_store = FAISS.load_local(
                    user_db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(q)
                
                if not docs:
                    return "No relevant info found in the documents."
                
                return "\n\n".join(
                    [f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
                )
            except Exception as e:
                return f"Retrieval Error: {str(e)}"

        tools = [document_retriever]

        # 3. YOUR EXACT SYSTEM PROMPT (Preserved 100%)
        system_prompt = (
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
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        # 4. Execute Agent with User's Specific Memory
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            memory=session_data["memory"], # Isolate conversation history
            max_iterations=4,
            handle_parsing_errors=True
        )

        try:
            response = agent_executor.invoke({"input": query})
            clean_text = format_clean_output(response["output"])
            return {"answer": clean_text}
        except Exception as e:
            logger.error(f"Agent Execution Crash: {e}")
            return {"answer": "I encountered a processing error. Please check the backend logs."}

# --- 4. GLOBAL INSTANCE (Singleton Pattern) ---
session_manager = SessionManager()

# --- 5. EXPOSED FUNCTIONS (Updated for Session Support) ---
def build_database(upload_dir, session_id="default"):
    return session_manager.ingest_documents(session_id, upload_dir)

def get_answer(query, session_id="default"):
    return session_manager.ask(session_id, query)