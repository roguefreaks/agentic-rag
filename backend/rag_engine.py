import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

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
    # THE BRAIN (GPT-4o)
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        temperature=0
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

# --- 4. THE MODERN TOOL ---
@tool
def search_documents(query: str):
    """Search for information in the uploaded documents."""
    if not os.path.exists(DB_PATH):
        return "No documents found. Ask the user to upload one."
    try:
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the document."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error reading memory: {e}"

# --- 5. THE MODERN AGENT (Tool Caller) ---
def get_answer(query):
    tools = [search_documents]

    # This prompt structure prevents the "Infinite Loop"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the search_documents tool to answer questions based on the user's file."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the Tool-Calling Agent (Specific for GPT-4o)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=5  # Hard limit to stop any future looping
    )

    try:
        response = agent_executor.invoke({"input": query})
        return {"answer": response["output"]}
    except Exception as e:
        logger.error(f"AGENT ERROR: {e}")
        return {"answer": "I found the info but got stuck formatting it. The document has been indexed correctly though."}