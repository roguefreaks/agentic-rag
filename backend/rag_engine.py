import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

# Load env vars
load_dotenv()

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. CRITICAL FIX: SANITIZE INPUTS ---
# This fixes the "DeploymentNotFound" 404 error by cleaning user mistakes
def get_clean_env(key):
    val = os.getenv(key)
    if not val:
        return ""
    # Remove quotes, spaces, and trailing slashes (which break Azure)
    return val.strip().strip('"').strip("'").rstrip('/')

DB_PATH = "faiss_index_web"
API_KEY = get_clean_env("AZURE_OPENAI_API_KEY")
ENDPOINT = get_clean_env("AZURE_OPENAI_ENDPOINT")
API_VERSION = get_clean_env("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

# Log connection details (Masked) for debugging
logger.info(f"Connecting to: {ENDPOINT}")
logger.info(f"Using Version: {API_VERSION}")

# --- 2. INITIALIZE MODELS ---
try:
    llm = AzureChatOpenAI(
        azure_deployment="o3-mini",  # Your deployment name
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        temperature=0  # Agents need precision
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small", # Your embedding name
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY
    )
except Exception as e:
    logger.error(f"CRITICAL MODEL ERROR: {e}")

def build_database(upload_dir):
    """Reads files and rebuilds the Vector Memory."""
    documents = []
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    if not documents:
        return "No documents found."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
    vector_store.save_local(DB_PATH)
    return f"Agent Memory Updated: {len(documents)} pages indexed."

# --- 3. THE AGENT TOOL (The "Hand" of the Agent) ---
def search_documents(query: str):
    """Searches the vector database for relevant info."""
    if not os.path.exists(DB_PATH):
        return "Error: No documents found. Tell the user to upload a file."
    
    try:
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Memory Access Error: {str(e)}"

# --- 4. THE AGENT BRAIN (The "Mind") ---
def get_answer(query):
    # Define the "Tools" the agent can use
    tools = [
        Tool(
            name="Document_Search",
            func=search_documents,
            description="Use this to look up information in the uploaded PDF/files. Always use this first."
        )
    ]

    # Initialize a REAL ReAct Agent (Reason+Act)
    # This creates the "Thought -> Action -> Observation" loop your friend wants
    agent_chain = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, # This logs the "Thinking" process
        handle_parsing_errors=True
    )

    try:
        # Run the agent loop
        response = agent_chain.invoke(query)
        
        # If the output is a dictionary (common in newer versions), extract 'output'
        if isinstance(response, dict) and "output" in response:
            return {"answer": response["output"]}
        return {"answer": str(response)}
        
    except Exception as e:
        logger.error(f"Agent Crash: {e}")
        # Detailed error for debugging
        return {"answer": f"Agent Error: {str(e)}. Check Render Logs for 'DeploymentNotFound'."}