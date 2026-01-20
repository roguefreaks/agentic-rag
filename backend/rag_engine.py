import os
import logging
import re
import shutil
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
DB_PATH = "faiss_index_web"
CHUNK_SIZE = 1200  # Increased for better context
CHUNK_OVERLAP = 200
MAX_HISTORY = 10   # Remember last 10 messages

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
    # Remove headers (###)
    text = re.sub(r'#+\s*', '', text)
    # Remove bold/italic (** or *)
    text = re.sub(r'\*\*|__', '', text)
    # Ensure proper spacing
    return text.strip()

# --- 3. THE RAG ENGINE CLASS (Enterprise Architecture) ---
class AgenticRAG:
    def __init__(self):
        self._init_environment()
        self._init_models()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        logger.info("AgenticRAG System Initialized Successfully.")

    def _init_environment(self):
        """Loads and validates all environment variables."""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.endpoint = clean_azure_url(os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
        
        if not self.api_key or not self.endpoint:
            logger.critical("Missing Azure Credentials! System will fail.")
            raise ValueError("Azure API Key and Endpoint are required.")

    def _init_models(self):
        """Initializes Azure Models with Retry Logic."""
        try:
            # The Brain: GPT-4o (Standard, Powerful, Robust)
            self.llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                temperature=0.0, # Precision is key
                max_retries=3
            )

            # The Eyes: Embeddings
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

    def ingest_documents(self, upload_dir: str) -> str:
        """
        Ingest Process:
        1. Validate Dir
        2. Load Files (PDF/Docx)
        3. Split Text
        4. Vectorize & Index
        """
        documents = []
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

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
                else:
                    logger.warning(f"Skipping unsupported file: {filename}")
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

        if not documents:
            return "No valid documents found to index."

        # 2. Advanced Text Splitting (Respects Sentence Boundaries)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # 3. Vector Database Creation
        try:
            vector_store = FAISS.from_documents(splits, self.embeddings)
            vector_store.save_local(DB_PATH)
            # Reset memory on new document upload so context is fresh
            self.memory.clear() 
            logger.info(f"Successfully indexed {len(documents)} pages.")
            return f"Indexed {len(documents)} pages. I am ready for questions."
        except Exception as e:
            logger.error(f"Vector Store Failure: {e}")
            raise RuntimeError(f"Indexing Failed: {e}")

    def get_search_tool(self):
        """Creates the retrieval tool with access to the Vector DB."""
        @tool
        def document_retriever(query: str):
            """
            Useful for finding specific information in the uploaded documents.
            Always use this tool first when asked about the candidate or file content.
            """
            if not os.path.exists(DB_PATH):
                return "The database is empty. Please ask the user to upload a document."
            
            try:
                # Safe Loading
                vector_store = FAISS.load_local(
                    DB_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)
                
                if not docs:
                    return "No relevant info found in the documents."
                
                # Format context clearly
                context = "\n\n".join(
                    [f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
                )
                return context
            except Exception as e:
                return f"Retrieval Error: {str(e)}"
        
        return document_retriever

    def ask(self, query: str) -> Dict[str, Any]:
        """
        The Main Agent Loop:
        1. Setup Tools
        2. Define System Persona (Professional, No Markdown)
        3. Execute Agent
        """
        tools = [self.get_search_tool()]

        # THE "META-LEVEL" SYSTEM PROMPT (REPLACED ONLY)
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
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            memory=self.memory,
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
rag_agent = AgenticRAG()

# --- 5. EXPOSED FUNCTIONS (For main.py to call) ---
def build_database(upload_dir):
    return rag_agent.ingest_documents(upload_dir)

def get_answer(query):
    return rag_agent.ask(query)
