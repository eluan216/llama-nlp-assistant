# LLM-Powered Document Q&A and Summarization System

An end-to-end Retrieval-Augmented Generation (RAG) system for document question-answering and summarization using open-source LLMs from Hugging Face.

## Features

- **Document Upload**: Support for PDF and TXT files
- **Intelligent Chunking**: Recursive character text splitting with overlap
- **Semantic Embeddings**: High-quality sentence embeddings via `sentence-transformers`
- **Vector Retrieval**: Fast similarity search with FAISS
- **Open-Source LLM**: Powered by Microsoft Phi-3-mini (or easily swappable models)
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

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/eluan216/llama-nlp-assistant.git
cd llama-nlp-assistant
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser. Upload a PDF or TXT file, ask questions, or request a summary.

> **Note**: The first run will download the embedding model (~90MB) and the LLM (~2.3GB for Phi-3-mini). Subsequent runs are much faster.

### 3. Hardware Recommendations

| Setup              | Recommended Model              | Notes                          |
|--------------------|--------------------------------|--------------------------------|
| CPU only           | `microsoft/Phi-3-mini-4k-instruct` | Works, but slower generation  |
| GPU (8GB+)         | Same or larger models          | Significantly faster           |
| Low memory         | Quantized models or smaller LLMs | See `src/llm_pipeline.py`     |

## How It Works (RAG Pipeline)

1. **Load** → Extract text from PDF/TXT
2. **Chunk** → Split into overlapping segments
3. **Embed** → Convert chunks to dense vectors
4. **Index** → Store in FAISS for fast retrieval
5. **Retrieve** → Find top-k relevant chunks for a query
6. **Generate** → LLM answers using retrieved context

## Configuration

Key parameters can be adjusted in the Streamlit sidebar or directly in the source modules:

- Chunk size / overlap
- Number of retrieved documents (`top_k`)
- Generation temperature / max tokens
- Model choice

## Evaluation

The system includes basic evaluation helpers in `src/evaluation.py`:

- Context relevance scoring
- Simple ROUGE-style overlap metrics
- Answer-context grounding score

## Extending the Project

- Swap the LLM in `src/llm_pipeline.py` (Gemma-2, Mistral, Llama-3.2, etc.)
- Add fine-tuning with PEFT/LoRA
- Replace FAISS with Chroma or Qdrant
- Add multi-document chat history
- Deploy with Docker + Hugging Face Spaces / Streamlit Cloud

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
