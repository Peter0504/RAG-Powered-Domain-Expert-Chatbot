# RAG-Powered-Domain-Expert-Chatbot
# 📚 Advanced RAG Document Assistant

This is a full-featured Retrieval-Augmented Generation (RAG) application that allows you to chat with your own local documents. It uses a hybrid search approach (combining semantic AI search with BM25 keyword matching) to find the most relevant information and cites its sources directly in the chat interface.

## ✨ Features
* **Multi-Format Document Processing:** Automatically extracts text from PDF, HTML, DOCX, TXT, and MD files.
* **Hybrid Search:** Combines ChromaDB vector embeddings (`sentence-transformers`) with BM25 keyword search for highly accurate retrieval.
* **Conversation Memory:** Remembers the last 10 back-and-forth exchanges to accurately answer follow-up questions.
* **Source Citations:** Displays the exact documents and text snippets used to generate the answer.
* **Interactive UI:** Built with Gradio for a clean, browser-based chat experience.

## 📋 Prerequisites
1. **Python:** Ensure you have Python 3.9 or higher installed on your system.
2. **OpenAI API Key:** You will need a valid OpenAI API key (`sk-...`) to generate the answers using the `gpt-4o-mini` model.

## 🛠️ Installation & Setup

**1. Clone or download the repository**
Ensure the main script is saved as `rag_app.py`.

**2. Install dependencies**
Open your terminal or command prompt, navigate to the project folder, and run:
```bash
pip install chromadb sentence-transformers gradio openai pymupdf python-docx beautifulsoup4 rank_bm25 nltk tiktoken numpy
