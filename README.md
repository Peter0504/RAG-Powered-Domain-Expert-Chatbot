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
```
## 🚀 How to Run the Application
You must set your OpenAI API key as an environment variable before running the script. Open your terminal or command prompt in the project folder and run the commands specific to your operating system:

### For Windows (Command Prompt):

```DOS
set OPENAI_API_KEY="sk-your-api-key-here"
python rag_app.py
```
### For Windows (PowerShell):

```PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"
python rag_app.py
```
### For Mac / Linux:

```Bash
export OPENAI_API_KEY="sk-your-api-key-here"
python rag_app.py
```
## 💻 Usage
On the first run, the system will read your documents folder, chunk the text, and build the ChromaDB and BM25 databases. This may take a few minutes depending on the number of files.

Once indexing is complete, a local web server will start.

Look for a URL in your terminal (usually http://127.0.0.0:7860 or similar) and open it in your web browser to start chatting with your documents!

## ⚙️ Advanced Configuration
If you want to tweak how the application works, open rag_app.py and modify the Config class near the top of the file. You can adjust chunk sizes, memory length, change the embedding model, or modify the hybrid search weights.
