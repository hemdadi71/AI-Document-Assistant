# AI Document Assistant (RAG-based LLM Application)

An AI-powered document assistant that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

The application retrieves relevant document information using semantic search and generates context-aware answers with Large Language Models (LLMs).

---

## ✨ Features

- 📄 PDF document upload and processing
- 🔍 Semantic search over documents
- 🧩 Text chunking and embedding generation
- 🧠 RAG-based question answering
- 📚 Multi-document knowledge base support
- 📌 Source citation for generated answers
- ⚡ Streaming LLM responses
- 💾 Persistent vector storage

---

## 🏗️ Architecture

PDF Upload => Text Extraction => Text Chunking => Embedding Generation => FAISS Vector Search =>Relevant Context Retrieval => LLM Generation => 


---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI

### AI / LLM
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- Embedding Models
- Groq API (OpenAI-compatible API)

### Vector Search
- FAISS

### Document Processing
- PyPDF

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>

cd AI-Document-Assistant

python -m venv .venv

.venv/Scripts/activate

pip install -r requirements.txt
```
Create a .env file:
GROQ_API_KEY=your_api_key_here

‍‍‍‍
```bash
uvicorn main:app --reload
```
API documentation:
http://localhost:8000/docs

Example Response:

```bash
{
  "answer": "His name is hosein emdadi",
  "sources": [
    {
      "source": "sample.pdf",
      "chunk_id": 0
    }
  ]
}
```

