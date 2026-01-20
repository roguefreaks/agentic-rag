# ⚡ Titanium RAG - Enterprise Edition v8.2

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Version](https://img.shields.io/badge/Version-v8.2-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20LangChain-009688)
![Frontend](https://img.shields.io/badge/Frontend-React%20%7C%20Titanium%20UI-61DAFB)

**Titanium RAG** is a high-availability, session-isolated Retrieval-Augmented Generation (RAG) system designed for secure enterprise document intelligence. It allows users to upload documents (PDF, DOCX, TXT) and chat with them using an **Azure OpenAI GPT-4o** agent.

The system is engineered with a **"Titanium" Architecture**, featuring strict security, structured telemetry, and a fault-tolerant "Forgiving Mode" for mobile compatibility.

---

## 🚀 Key Features

### 🛡️ Enterprise Security
* **Magic Byte Validation:** preventing file spoofing by inspecting hex signatures.
* **Session Isolation:** Every user gets a unique, ephemeral Vector DB. Data never leaks between sessions.
* **Rate Limiting:** Token-bucket algorithm (100 req/min) to prevent abuse.
* **Security Headers:** X-Content-Type-Options, X-Frame-Options, and XSS protection.

### 🧠 Advanced Cognitive Engine
* **Polyglot Ingestion:** Supports PDF, DOCX, and TXT files.
* **Recursive Semantic Splitting:** Context-aware chunking for better retrieval accuracy.
* **Agentic Reasoning:** Uses GPT-4o with Tool Calling to "search" documents before answering.
* **Strict Factuality:** System prompts enforce "No Hallucination" rules.

### 📱 Robust Frontend (Titanium UI)
* **Mobile-First Design:** Responsive sidebar and overlay menus.
* **Forgiving Uploads:** Custom Axios implementation to handle mobile browser "boundary" bugs (Fixed v8.2).
* **Real-time Feedback:** Toast notification system and typing indicators.

---

## 🏗️ Architecture Overview

The system is split into two microservices:

### 1. Backend (`/backend`)
* **Framework:** FastAPI (Python 3.10+)
* **Core Engine:** `rag_engine.py` (Singleton Pattern)
* **Vector Store:** FAISS (Local Ephemeral Storage)
* **Models:** Azure OpenAI (GPT-4o + text-embedding-3-small)
* **Observability:** JSON Structured Logging

### 2. Frontend (`/frontend`)
* **Framework:** React 18
* **HTTP Client:** Axios (Configured for multipart/form-data)
* **Styling:** Custom CSS Variables (Dark Mode native)

---

## 🛠️ Local Installation & Setup

### Prerequisites
* Python 3.10+
* Node.js 16+
* Azure OpenAI API Credentials

### 1. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
touch .env
