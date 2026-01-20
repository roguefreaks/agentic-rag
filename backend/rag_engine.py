import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load env vars
load_dotenv()

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (WITH CLEANING) ---
# We use .strip() to remove accidental spaces or quotes from Render
def get_clean_env(key):
    val = os.getenv(key)
    if val:
        return val.strip().strip('"').strip("'")
    return None

DB_PATH = "faiss_index_web"
API_KEY = get_clean_env("AZURE_OPENAI_API_KEY")
ENDPOINT = get_clean_env("AZURE_OPENAI_ENDPOINT")
API_VERSION = get_clean_env("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

# Log the config (Masked) to help debug
logger.info(f"--- SYSTEM STARTUP ---")
logger.info(f"Endpoint: {ENDPOINT}")
logger.info(f"Version: {API_VERSION}")
logger.info(f"Key Loaded: {'Yes' if API_KEY else 'No'}")

# MODELS
try:
    llm = AzureChatOpenAI(
        azure_deployment="o3-mini", # Must match your screenshot EXACTLY
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small", # Must match your screenshot EXACTLY
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY
    )
except Exception as e:
    logger.error(f"Model Init Error: {e}")

def build_database(upload_dir):
    """Reads all files and rebuilds the index."""
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

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Save Index
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
    vector_store.save_local(DB_PATH)
    return f"Indexed {len(documents)} pages."

def get_answer(query):
    """Directly retrieves context and asks the LLM."""
    if not os.path.exists(DB_PATH):
        return {"answer": "System is empty. Please upload a document."}

    try:
        # 1. Load Memory
        vector_store = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 2. Find relevant text
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        
        if not docs:
            return {"answer": "I couldn't find any relevant info in the uploaded file."}
            
        # 3. Combine text
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 4. Send to AI
        system_msg = "You are a precise assistant. Answer the question based ONLY on the provided Context."
        user_msg = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        messages = [("system", system_msg), ("user", user_msg)]
        
        response = llm.invoke(messages)
        return {"answer": response.content}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"answer": f"System Error: {str(e)}"}