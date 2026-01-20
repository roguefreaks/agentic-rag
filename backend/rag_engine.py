import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

# Load env vars (Local) or use System vars (Render)
load_dotenv()

# LOGGING SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURATION
DB_PATH = "faiss_index_web"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# INITIALIZE MODELS
llm = AzureChatOpenAI(
    azure_deployment="o3-mini",
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY
)

def build_database(upload_dir):
    """Reads all files, chunks them, and rebuilds the FAISS index."""
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
        return "No documents found to index."

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create & Save Vector Store
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
    vector_store.save_local(DB_PATH)
    return f"Indexed {len(documents)} pages."

def get_answer(query):
    """Loads the Latest Index and asks the AI."""
    # 1. CHECK IF DATABASE EXISTS
    if not os.path.exists(DB_PATH):
        return {"answer": "System is empty. Please upload a document first."}

    # 2. FORCE RELOAD THE INDEX (This fixes the bug!)
    # allow_dangerous_deserialization is required for newer LangChain versions
    try:
        vector_store = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
    except Exception as e:
        logger.error(f"Index load error: {e}")
        return {"answer": "Memory corrupted. Please reset/re-upload."}

    # 3. SETUP RETRIEVER
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 4. DEFINE TOOL
    retriever_tool = Tool(
        name="document_search",
        func=retriever.invoke,
        description="Search for information in the uploaded documents."
    )
    
    tools = [retriever_tool]

    # 5. CREATE AGENT
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the document_search tool to find information. If the info is not in the docs, say so."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 6. EXECUTE
    try:
        response = agent_executor.invoke({"input": query})
        return {"answer": response["output"]}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"answer": "I encountered an error processing that request."}