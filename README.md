# LLM-Powered Document Q&A and Summarization System

An end-to-end Retrieval-Augmented Generation (RAG) system for document question-answering and summarization using open-source LLMs from Hugging Face.

## Live Demo

Try the deployed app on Streamlit Cloud:

**https://llama-nlp-assistant-4hyjezelnvzdrndrsxqjts.streamlit.app**

> First load may take 1–2 minutes while the model downloads. After that, Q&A and summarization run in the browser UI.

## Features

- **Document Upload**: Support for PDF and TXT files
- **Intelligent Chunking**: Recursive character text splitting with overlap
- **Semantic Embeddings**: High-quality sentence embeddings via `sentence-transformers`
- **Vector Retrieval**: Fast similarity search with FAISS
- **Open-Source LLM**: Default `HuggingFaceTB/SmolLM2-360M-Instruct` (fits Streamlit Cloud free tier)
- **Question Answering**: Context-aware answers grounded in your documents
- **Summarization**: Generate concise summaries of uploaded documents
- **Simple Evaluation**: Basic relevance and response quality metrics
- **Clean Streamlit UI**: Interactive web interface

## Project Structure

```
llama-nlp-assistant/
├── data/                  # Sample data & uploads
├── src/
│   ├── data_loader.py    # PDF/TXT loading
│   ├── chunking.py       # Text chunking
│   ├── embeddings.py     # Embedding model
│   ├── retriever.py      # FAISS vector store + retrieval
│   ├── llm_pipeline.py   # LLM generation pipeline
│   └── evaluation.py     # Simple evaluation metrics
├── notebooks/            # Experimentation notebooks
├── app.py                # Streamlit application
├── requirements.txt
└── README.md
```

## Quick Start (local)

```bash
git clone https://github.com/eluan216/llama-nlp-assistant.git
cd llama-nlp-assistant
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Deploy `eluan216/llama-nlp-assistant`
4. Set **Main file path** to `app.py` (not `streamlit_app.py`)
5. Click **Deploy**

After you push new commits to `main`, use **Manage app → Reboot** so Streamlit picks up changes.

## How It Works (RAG Pipeline)

1. **Load** → Extract text from PDF/TXT
2. **Chunk** → Split into overlapping segments
3. **Embed** → Convert chunks to dense vectors
4. **Index** → Store in FAISS for fast retrieval
5. **Retrieve** → Find top-k relevant chunks for a query
6. **Generate** → LLM answers using retrieved context

## Tech Stack

- **Python 3.10+**
- **PyTorch** + **Hugging Face Transformers**
- **sentence-transformers**
- **FAISS**
- **Streamlit**
- **pypdf**

## License

MIT

---

Built as a portfolio project demonstrating practical LLM + RAG engineering skills.
