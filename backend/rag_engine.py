import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType, Tool

# Load Environment Variables
load_dotenv()

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. NUCLEAR URL CLEANING ---
def clean_url(url):
    if not url: return ""
    url = url.strip().strip('"').strip("'")
    url = url.rstrip('/')
    if "/openai" in url:
        url = url.split("/openai")[0]
    return url

DB_PATH = "faiss_index_web"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
ENDPOINT = clean_url(os.getenv("AZURE_OPENAI_ENDPOINT"))
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()

# --- 2. INITIALIZE MODELS ---
try:
    # THE BRAIN (o3-mini) - REMOVED TEMPERATURE TO FIX CRASH
    llm = AzureChatOpenAI(
        azure_deployment="o3-mini",
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        # temperature=0  <-- DELETED THIS LINE. o3-mini DOES NOT ALLOW IT.
    )

    # THE EYES (Embeddings)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    )
except Exception as e:
    logger.error(f"CRITICAL MODEL INIT FAILURE: {e}")

# --- 3. DATABASE BUILDER ---
def build_database(upload_dir):
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
            logger.error(f"Failed to read {filename}: {e}")

    if not documents:
        return "No documents found."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    try:
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        vector_store.save_local(DB_PATH)
        return f"Agent Memory Updated: {len(documents)} pages indexed."
    except Exception as e:
        logger.error(f"EMBEDDING FAILURE: {e}")
        raise Exception(f"Azure Connection Failed: {str(e)}")

# --- 4. AGENT TOOL ---
def search_documents(query: str):
    if not os.path.exists(DB_PATH):
        return "System Memory Empty. Tell the user to upload a file."
    try:
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Memory Read Error: {e}"

# --- 5. THE REACT AGENT ---
def get_answer(query):
    tools = [
        Tool(
            name="Document_Search",
            func=search_documents,
            description="Use this tool to find facts in the uploaded document."
        )
    ]

    try:
        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        response = agent_chain.invoke(query)
        
        if isinstance(response, dict) and "output" in response:
            return {"answer": response["output"]}
        return {"answer": str(response)}
        
    except Exception as e:
        logger.error(f"AGENT CRASH: {e}")
        return {"answer": f"Agent Error: {str(e)}"}