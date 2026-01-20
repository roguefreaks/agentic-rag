import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- SAFE IMPORT FOR AGENT EXECUTOR ---
# This fixes the "ImportError" by trying both locations
from langchain.agents import AgentExecutor

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ... (Rest of your code stays the same)

# --- ⚠️ PASTE YOUR KEYS HERE ⚠️ ---

# --- ⚠️ PASTE YOUR KEYS HERE ⚠️ ---

# 1. DETAILS FOR THE BRAIN (o3-mini)
# DELETE YOUR REAL KEY FROM HERE!
CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
CHAT_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = "o3-mini"

# 2. DETAILS FOR THE EYES (text-embedding)
# DELETE YOUR REAL KEY FROM HERE TOO!
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBED_DEPLOYMENT = "text-embedding-3-small"

API_VERSION = "2024-12-01-preview"

# --- SYSTEM SETUP ---
DB_PATH = "faiss_index_web"

# Initialize Embedding Model (The Eyes)
try:
    EMBEDDING_MODEL = AzureOpenAIEmbeddings(
        azure_deployment=EMBED_DEPLOYMENT,
        openai_api_version=API_VERSION,
        azure_endpoint=EMBED_ENDPOINT,
        api_key=EMBED_KEY
    )
except Exception as e:
    logger.error(f"Error initializing Embeddings: {e}")

# --- HELPER FUNCTIONS ---
def create_retriever_tool(retriever, name: str, description: str):
    def retrieve_and_format(query: str):
        try:
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return "Error retrieving documents."
    return Tool(name=name, description=description, func=retrieve_and_format)

def load_doc(file_path):
    ext = file_path.split(".")[-1].lower()
    try:
        if ext == "pdf": return PyPDFLoader(file_path)
        elif ext == "txt": return TextLoader(file_path, encoding='utf-8')
        elif ext == "docx": return Docx2txtLoader(file_path)
        elif ext == "csv": return CSVLoader(file_path)
        else: return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

# --- FAST UPLOAD ---
def build_database(folder_path):
    all_docs = []
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    for filename in os.listdir(folder_path):
        if filename.startswith("."): continue
        file_path = os.path.join(folder_path, filename)
        loader = load_doc(file_path)
        if loader:
            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

    if not all_docs: return "No valid documents found."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    logger.info(f"Processing {len(chunks)} chunks...")

    try:
        vector_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)
        vector_store.save_local(DB_PATH)
        return f"Success! Indexed {len(chunks)} chunks."
    except Exception as e:
        logger.error(f"Azure Embedding Error: {e}")
        return f"Error: {e}"

# --- AGENTIC LOGIC ---
def get_answer(query):
    if not os.path.exists(DB_PATH):
        return {"result": "⚠️ System offline. Please upload documents first."}

    try:
        vector_store = FAISS.load_local(DB_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        tool = create_retriever_tool(
            retriever,
            "search_resume_and_docs",
            "Searches user documents."
        )
        tools = [tool]

        # Use Azure o3-mini (The Brain)
        llm = AzureChatOpenAI(
            azure_deployment=CHAT_DEPLOYMENT,
            openai_api_version=API_VERSION,
            azure_endpoint=CHAT_ENDPOINT,
            api_key=CHAT_KEY,
            temperature=1 
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the 'search_resume_and_docs' tool to answer questions."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        response = agent_executor.invoke({"input": query})
        
        return {"answer": response['output'], "sources": ["Azure o3-mini"]}

    except Exception as e:
        logger.error(f"Agent Error: {e}")
        return {"answer": f"An error occurred: {str(e)}", "sources": []}